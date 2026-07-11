#!/usr/bin/env ruby
# frozen_string_literal: true

# Verify public-course facts, homepage narrative, discoverability, and the
# rendered GitHub Pages contract without calling external model APIs.
#
#   ruby scripts/check-showcase.rb
#   ruby scripts/check-showcase.rb --built _site

require "json"
require "pathname"
require "yaml"

ROOT = Pathname.new(__dir__).join("..").expand_path
ERRORS = []

def check(condition, message)
  ERRORS << message unless condition
end

def read(relative_path)
  File.read(ROOT.join(relative_path), encoding: "UTF-8")
end

def meta_content(html, key)
  escaped = Regexp.escape(key)
  html[/<meta\b[^>]*(?:name|property)=["']#{escaped}["'][^>]*content=["']([^"']*)["'][^>]*>/i, 1]
end

def png_dimensions(path)
  bytes = File.binread(path, 24)
  return nil unless bytes.start_with?("\x89PNG\r\n\x1a\n".b)

  bytes[16, 8].unpack("NN")
end

readme = read("README.md")
layout = read("_layouts/default.html")
theme = read("_sass/opendesign/theme.scss")
config = YAML.safe_load(read("_config.yml"), aliases: true) || {}
workflow = read(".github/workflows/pages.yml")
smoke = read("scripts/smoke-test.sh")
concepts = read("docs/concepts.md")
challenges = read("docs/challenges.md")
prompts = read("docs/prompts-cheatsheet.md")
debug_recipes = read("docs/debug-recipes.md")

# Repository-derived facts used on the homepage.
lesson_count = Dir.glob(ROOT.join("tutorial/week-*/*.md")).count do |path|
  File.basename(path) != "README.md"
end
all_files_block = smoke[/ALL_FILES=\((.*?)^\)/m, 1].to_s
entrypoint_count = all_files_block.scan(/["'](final\/[^"']+\.py)["']/).flatten.uniq.length
concept_count = concepts.scan(/^###\s+\d+\./).length
challenge_count = challenges.scan(/^##\s+\d+\./).length
prompt_count = prompts.scan(/^###\s+模板\s+/).length
debug_count = debug_recipes.scan(/^###\s+\d+\.\d+/).length

expected = {
  "lessons" => [lesson_count, 16],
  "entrypoints" => [entrypoint_count, 14],
  "concepts" => [concept_count, 18],
  "challenges" => [challenge_count, 7],
  "prompt templates" => [prompt_count, 19],
  "debug recipes" => [debug_count, 18]
}
expected.each do |label, (actual, target)|
  check(actual == target, "#{label}: expected #{target}, found #{actual}")
end

{
  "lessons" => lesson_count,
  "entrypoints" => entrypoint_count,
  "concepts" => concept_count,
  "challenges" => challenge_count
}.each do |metric, value|
  check(readme.include?(%{data-metric="#{metric}">#{value}<}), "homepage metric #{metric}=#{value} is missing")
end

# Public narrative contract.
{
  "problem" => 'id="problem"',
  "learning system" => 'id="learning-system"',
  "curriculum" => 'id="curriculum"',
  "evidence" => 'id="evidence"',
  "role and AI boundary" => 'id="role"',
  "limitations" => 'id="limitations"',
  "English summary" => "English summary.",
  "historical run boundary" => "12 PASS / 1 PARTIAL / 1 SKIP"
}.each do |label, marker|
  check(readme.include?(marker), "homepage is missing #{label}: #{marker}")
end

check(readme.scan(/<h1\b/i).length == 1, "homepage source must contain exactly one h1")
check(readme.include?("本次前端重构没有把它们重新宣称为通过"), "homepage must distinguish historical API runs from current checks")
check(readme.include?("Human owns") && readme.include?("AI assists"), "homepage must disclose the human/AI responsibility boundary")
check(!readme.include?("4 周 14 篇"), "homepage still contains the stale 14-lesson claim")
check(!readme.include?("21 个高频 prompt"), "homepage still contains the stale 21-template claim")
check(!readme.include?("16 个高频报错"), "homepage still contains the stale 16-recipe claim")

# Discoverability, identity, design-system, and accessible navigation.
global_urls = [
  "https://estelledc.github.io/",
  "https://estelledc.github.io/about/",
  "https://estelledc.github.io/resume/",
  "https://github.com/estelledc"
]
global_urls.each do |url|
  check(layout.include?(url), "layout is missing global identity URL: #{url}")
end

check(layout.include?('class="jx-skip-link"'), "layout is missing the skip link")
check(layout.include?('class="site-nav-menu"'), "layout is missing accessible mobile navigation")
check(layout.include?('twitter:description'), "layout must emit twitter:description")
check(layout.include?('"@type": "Course"'), "homepage JSON-LD must describe a Course")
check(layout.include?('"@type": "Person"'), "homepage JSON-LD must describe its creator")
check(theme.include?("@media (prefers-reduced-motion: reduce)"), "theme is missing reduced-motion handling")
check(theme.include?("@media (max-width: 560px)"), "theme is missing the mobile breakpoint")
check(read("_sass/jx/VERSION").strip == "2.0.0", "Jason DS vendor version must be 2.0.0")
check(read("_sass/jx/_tokens.scss").include?("Tokens v2.0.0"), "compiled Sass partial is not using Jason DS v2 tokens")
%w[tokens base components].each do |bundle|
  check(read("_sass/jx/_#{bundle}.scss") == read("_sass/jx/#{bundle}.css"), "Jason DS Sass partial drifted from synced #{bundle}.css")
end

check(config["url"] == "https://estelledc.github.io", "canonical host is unexpected")
check(config["baseurl"] == "/langchain-langgraph-langsmith-tutorial", "baseurl is unexpected")
check(config["locale"] == "zh_CN", "locale must be zh_CN")
check(config.dig("author", "name") == "Jason Xun", "author must be Jason Xun")
check(config["repository"] == "estelledc/langchain-langgraph-langsmith-tutorial", "repository metadata is missing")

check(workflow.include?("ruby scripts/check-showcase.rb --built _site"), "Pages workflow must check the rendered showcase")
check(workflow.include?("python3 -m compileall -q final"), "Pages workflow must compile-check Python references")
check(workflow.include?("bundle exec htmlproofer _site"), "Pages workflow must check internal links")
check(workflow.scan(%r{uses:\s+[^\s@]+@([0-9a-f]{40})}).length == workflow.scan(/^\s*uses:/).length, "Pages workflow actions must be pinned to immutable commit SHAs")

og_path = ROOT.join("assets/og-tutorial-zero.png")
check(og_path.file?, "social preview is missing: assets/og-tutorial-zero.png")
if og_path.file?
  check(png_dimensions(og_path) == [1200, 630], "social preview must be a 1200x630 PNG")
end

favicon_path = ROOT.join("assets/favicon.png")
check(favicon_path.file?, "favicon is missing: assets/favicon.png")
if favicon_path.file?
  check(png_dimensions(favicon_path) == [128, 128], "favicon must be a 128x128 PNG")
end

# Optional rendered output.
built_dir = nil
ARGV.each_with_index do |argument, index|
  built_dir = ARGV[index + 1] if argument == "--built"
end

if ARGV.include?("--built")
  check(built_dir && !built_dir.start_with?("--"), "--built requires a directory")

  if built_dir && !built_dir.start_with?("--")
    built_root = Pathname.new(built_dir)
    built_root = ROOT.join(built_root) unless built_root.absolute?
    built_index = built_root.join("index.html")
    check(built_index.file?, "built homepage does not exist: #{built_index}")

    if built_index.file?
      html = File.read(built_index, encoding: "UTF-8")
      canonical = html[/<link\b[^>]*rel=["']canonical["'][^>]*href=["']([^"']+)["'][^>]*>/i, 1]
      og_image = meta_content(html, "og:image")
      twitter_description = meta_content(html, "twitter:description")

      check(canonical == "https://estelledc.github.io/langchain-langgraph-langsmith-tutorial/", "rendered canonical URL is missing or incorrect")
      check(meta_content(html, "og:locale") == "zh_CN", "rendered og:locale must be zh_CN")
      check(og_image == "https://estelledc.github.io/langchain-langgraph-langsmith-tutorial/assets/og-tutorial-zero.png", "rendered og:image is missing or incorrect")
      check(meta_content(html, "twitter:card") == "summary_large_image", "rendered twitter card must use the large image")
      check(twitter_description && !twitter_description.strip.empty?, "rendered twitter:description is missing or empty")
      check(meta_content(html, "og:image:width") == "1200" && meta_content(html, "og:image:height") == "630", "rendered social image dimensions are missing or incorrect")
      check(!meta_content(html, "og:image:alt").to_s.strip.empty?, "rendered og:image:alt is missing")
      check(!meta_content(html, "twitter:image:alt").to_s.strip.empty?, "rendered twitter:image:alt is missing")
      check(html.scan(/<h1\b/i).length == 1, "rendered homepage must contain exactly one h1")
      check(html.include?('lang="en"') && html.include?("English summary."), "rendered homepage is missing its English summary")
      check(html.include?('id="content"') && html.include?('href="#content"'), "rendered homepage is missing its skip-link target")
      check(!html.include?("class=\"page-header\""), "legacy Cayman page header leaked into the homepage")

      global_urls.each do |url|
        check(html.include?(%{href="#{url}"}), "rendered homepage is missing identity URL: #{url}")
      end

      json_ld = html.scan(%r{<script\b[^>]*type=["']application/ld\+json["'][^>]*>(.*?)</script>}mi).flatten
      parsed = json_ld.each_with_object([]) do |payload, documents|
        documents << JSON.parse(payload)
      rescue JSON::ParserError => error
        ERRORS << "invalid rendered JSON-LD: #{error.message}"
      end
      graph_nodes = parsed.flat_map do |node|
        node.is_a?(Hash) && node["@graph"].is_a?(Array) ? node["@graph"] : [node]
      end
      check(graph_nodes.any? { |node| node.is_a?(Hash) && node["@type"] == "Course" }, "rendered JSON-LD is missing Course")
      check(graph_nodes.any? { |node| node.is_a?(Hash) && node["@type"] == "Person" && node["name"] == "Jason Xun" }, "rendered JSON-LD is missing creator Person")

      forbidden = ["/Users/", "/private/tmp/", "/tmp/showcase", "sk-你的key"]
      forbidden.each do |marker|
        check(!html.include?(marker), "rendered homepage exposes forbidden marker: #{marker}")
      end
    end
  end
end

if ERRORS.empty?
  mode = built_dir ? "source + rendered homepage" : "source"
  puts "check-showcase: OK (#{mode}; #{lesson_count} lessons, #{entrypoint_count} entrypoints, #{concept_count} concepts, #{challenge_count} challenges)"
  exit 0
end

ERRORS.each { |message| warn "SHOWCASE: #{message}" }
warn "\ncheck-showcase: #{ERRORS.length} failure(s)"
exit 1
