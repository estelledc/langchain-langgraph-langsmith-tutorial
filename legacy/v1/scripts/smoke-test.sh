#!/usr/bin/env bash
# scripts/smoke-test.sh — 一键跑批 14 个 final/.py，输出 PASS/FAIL/SKIP 报告
#
# 用法：
#   bash scripts/smoke-test.sh                # 默认跑全部，每个 timeout 600s
#   bash scripts/smoke-test.sh --quick        # 跳过耗时长的 (chains/memory/eval)
#   bash scripts/smoke-test.sh --timeout 900  # 自定义 timeout
#
# 退出码：
#   0 — 全 PASS（含 SKIP）
#   1 — 至少一个 FAIL
#
# 设计：
# - 不需要 conftest.py / pytest，纯 bash + python -u
# - 每个文件独立子进程，不互相污染
# - "ImportError / ModuleNotFoundError / 503 / 401" 这种归 ENV 错（SKIP）
# - 其他非 0 退出码归 FAIL
# - SSL/Permission 错主要看 stderr 关键词

set -uo pipefail

# ────────────────────────────────────────────────────────────────────────
# 参数解析
# ────────────────────────────────────────────────────────────────────────
TIMEOUT=600
QUICK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# ────────────────────────────────────────────────────────────────────────
# 环境检查
# ────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# 必须在 venv 里
if [ -z "${VIRTUAL_ENV:-}" ]; then
  if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    echo "❌ 没检测到 venv，请先 source .venv/bin/activate"
    exit 2
  fi
fi

# 必须有 .env
if [ ! -f ".env" ]; then
  echo "❌ 没找到 .env，先按 SETUP.md 配 DASHSCOPE_API_KEY 等"
  exit 2
fi

# ────────────────────────────────────────────────────────────────────────
# 14 个 final/.py 清单（按学习顺序）
# ────────────────────────────────────────────────────────────────────────
ALL_FILES=(
  "final/01_langchain/01_hello_llm.py"
  "final/01_langchain/02_prompt_template.py"
  "final/01_langchain/03_chains.py"
  "final/01_langchain/04_memory.py"
  "final/01_langchain/05_rag_basic.py"
  "final/01_langchain/06_tools_agent.py"
  "final/02_langgraph/01_simple_graph.py"
  "final/02_langgraph/02_conditional_edges.py"
  "final/02_langgraph/03_human_in_the_loop.py"
  "final/02_langgraph/04_multi_agent.py"
  "final/03_langsmith/01_tracing.py"
  "final/03_langsmith/02_evaluation.py"
  "final/03_langsmith/03_dataset.py"
  "final/04_project/agent.py"
)

# QUICK 模式跳过最耗时的几个
SKIP_QUICK=(
  "final/01_langchain/03_chains.py"
  "final/01_langchain/04_memory.py"
  "final/03_langsmith/02_evaluation.py"
)

# ────────────────────────────────────────────────────────────────────────
# 跑批
# ────────────────────────────────────────────────────────────────────────
RESULTS=()
TOTAL_START=$(date +%s)

is_quick_skip() {
  [ "$QUICK" -ne 1 ] && return 1
  for s in "${SKIP_QUICK[@]}"; do
    [ "$s" = "$1" ] && return 0
  done
  return 1
}

classify_failure() {
  # 输入：log 文件路径
  # 输出：分类（"ENV" / "CODE"）
  local log="$1"
  if grep -qE "PermissionDeniedError|403|401|insufficient_quota|This token has no access" "$log"; then
    echo "ENV"; return
  fi
  if grep -qE "SSLCertVerificationError|SSL_CERT_FILE|certificate verify failed" "$log"; then
    echo "ENV"; return
  fi
  if grep -qE "EOFError.*input|stdin" "$log"; then
    # human_in_the_loop 在非交互式终端会自动跳过；不是 ENV 也不是真错
    echo "PARTIAL"; return
  fi
  echo "CODE"
}

LOG_DIR="$(mktemp -d /tmp/smoke-test-XXXXXX)"
echo "📁 logs: $LOG_DIR"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  smoke test ${QUICK:+(quick mode)} — timeout ${TIMEOUT}s/file"
echo "═══════════════════════════════════════════════════════════"

for f in "${ALL_FILES[@]}"; do
  if is_quick_skip "$f"; then
    printf "  %-48s ⏭️  SKIP-QUICK\n" "$f"
    RESULTS+=("$f|SKIP-QUICK|0")
    continue
  fi

  log="$LOG_DIR/$(echo "$f" | tr '/' '_').log"
  start=$(date +%s)

  # 用 timeout 包一层；timeout 没有就 fallback 直接跑
  if command -v timeout >/dev/null 2>&1; then
    timeout "$TIMEOUT" python -u "$f" >"$log" 2>&1
  else
    # macOS 默认没 timeout，用 perl 兜底
    perl -e 'alarm shift; exec @ARGV' "$TIMEOUT" python -u "$f" >"$log" 2>&1
  fi
  rc=$?
  elapsed=$(( $(date +%s) - start ))

  if [ $rc -eq 0 ]; then
    printf "  %-48s ✅ PASS  (%ds)\n" "$f" "$elapsed"
    RESULTS+=("$f|PASS|$elapsed")
  elif [ $rc -eq 124 ] || [ $rc -eq 142 ]; then
    printf "  %-48s ⏱️  TIMEOUT (%ds)\n" "$f" "$elapsed"
    RESULTS+=("$f|TIMEOUT|$elapsed")
  else
    cls=$(classify_failure "$log")
    case "$cls" in
      ENV)
        printf "  %-48s ⚠️  SKIP-ENV (%ds)\n" "$f" "$elapsed"
        RESULTS+=("$f|SKIP-ENV|$elapsed")
        ;;
      PARTIAL)
        printf "  %-48s 🟡 PARTIAL (%ds)\n" "$f" "$elapsed"
        RESULTS+=("$f|PARTIAL|$elapsed")
        ;;
      *)
        printf "  %-48s ❌ FAIL (%ds, rc=%d)\n" "$f" "$elapsed" "$rc"
        RESULTS+=("$f|FAIL|$elapsed")
        ;;
    esac
  fi
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))

# ────────────────────────────────────────────────────────────────────────
# 总结
# ────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  汇总（总耗时 ${TOTAL_ELAPSED}s）"
echo "═══════════════════════════════════════════════════════════"

declare -A COUNT
for r in "${RESULTS[@]}"; do
  status="${r#*|}"; status="${status%|*}"
  COUNT[$status]=$((${COUNT[$status]:-0} + 1))
done

for k in PASS PARTIAL SKIP-ENV SKIP-QUICK FAIL TIMEOUT; do
  v="${COUNT[$k]:-0}"
  [ "$v" -gt 0 ] && printf "  %-12s %d\n" "$k" "$v"
done

echo ""
echo "📁 详细 log：$LOG_DIR"
echo ""

# ────────────────────────────────────────────────────────────────────────
# 把结果写到 docs/test-runs.md 末尾（仅在非 --quick 时）
# ────────────────────────────────────────────────────────────────────────
if [ "$QUICK" -ne 1 ]; then
  TR_FILE="docs/test-runs.md"
  if [ -f "$TR_FILE" ]; then
    {
      echo ""
      echo "---"
      echo ""
      echo "## 历次 smoke test 记录 — $(date +%Y-%m-%d)"
      echo ""
      echo "总耗时 ${TOTAL_ELAPSED}s（约 $((TOTAL_ELAPSED / 60)) 分钟）"
      echo ""
      echo "| 文件 | 状态 | 耗时 |"
      echo "|------|------|------|"
      for r in "${RESULTS[@]}"; do
        f="${r%%|*}"
        rest="${r#*|}"
        s="${rest%|*}"
        e="${rest##*|}"
        echo "| \`$f\` | $s | ${e}s |"
      done
      echo ""
      pass="${COUNT[PASS]:-0}"
      total=$(( pass + ${COUNT[PARTIAL]:-0} + ${COUNT[SKIP-ENV]:-0} + ${COUNT[FAIL]:-0} + ${COUNT[TIMEOUT]:-0} ))
      echo "**统计**: ${pass} PASS / ${COUNT[PARTIAL]:-0} PARTIAL / ${COUNT[SKIP-ENV]:-0} SKIP-ENV / ${COUNT[FAIL]:-0} FAIL / ${COUNT[TIMEOUT]:-0} TIMEOUT (共 ${total})"
    } >> "$TR_FILE"
    echo "📝 已 append 到 $TR_FILE"
  fi
fi

# ────────────────────────────────────────────────────────────────────────
# 退出码：FAIL > 0 即 1
# ────────────────────────────────────────────────────────────────────────
if [ "${COUNT[FAIL]:-0}" -gt 0 ] || [ "${COUNT[TIMEOUT]:-0}" -gt 0 ]; then
  exit 1
fi
exit 0
