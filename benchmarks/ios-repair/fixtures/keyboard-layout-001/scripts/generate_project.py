#!/usr/bin/env python3
"""Generate the deterministic Xcode project committed with this fixture."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "KeyboardFixture.xcodeproj"

PBXPROJ = r"""// !$*UTF8*$!
{
	archiveVersion = 1;
	classes = {};
	objectVersion = 56;
	objects = {

/* Begin PBXBuildFile section */
		B00000000000000000000001 /* AppDelegate.swift in Sources */ = {isa = PBXBuildFile; fileRef = D00000000000000000000001 /* AppDelegate.swift */; };
		B00000000000000000000002 /* KeyboardFixtureViewController.swift in Sources */ = {isa = PBXBuildFile; fileRef = D00000000000000000000002 /* KeyboardFixtureViewController.swift */; };
		B00000000000000000000003 /* KeyboardHeightResolver.swift in Sources */ = {isa = PBXBuildFile; fileRef = D00000000000000000000003 /* KeyboardHeightResolver.swift */; };
		B00000000000000000000004 /* UIKit.framework in Frameworks */ = {isa = PBXBuildFile; fileRef = D00000000000000000000030 /* UIKit.framework */; };
		B00000000000000000000011 /* ExistingBehaviorTests.swift in Sources */ = {isa = PBXBuildFile; fileRef = D00000000000000000000011 /* ExistingBehaviorTests.swift */; };
		B00000000000000000000012 /* XCTest.framework in Frameworks */ = {isa = PBXBuildFile; fileRef = D00000000000000000000031 /* XCTest.framework */; };
		B00000000000000000000021 /* KeyboardTaskOracleTests.swift in Sources */ = {isa = PBXBuildFile; fileRef = D00000000000000000000021 /* KeyboardTaskOracleTests.swift */; };
		B00000000000000000000022 /* XCTest.framework in Frameworks */ = {isa = PBXBuildFile; fileRef = D00000000000000000000031 /* XCTest.framework */; };
/* End PBXBuildFile section */

/* Begin PBXContainerItemProxy section */
		E00000000000000000000001 = {isa = PBXContainerItemProxy; containerPortal = A00000000000000000000010 /* Project object */; proxyType = 1; remoteGlobalIDString = A00000000000000000000001; remoteInfo = KeyboardFixture; };
		E00000000000000000000002 = {isa = PBXContainerItemProxy; containerPortal = A00000000000000000000010 /* Project object */; proxyType = 1; remoteGlobalIDString = A00000000000000000000001; remoteInfo = KeyboardFixture; };
/* End PBXContainerItemProxy section */

/* Begin PBXFileReference section */
		D00000000000000000000001 /* AppDelegate.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = AppDelegate.swift; sourceTree = "<group>"; };
		D00000000000000000000002 /* KeyboardFixtureViewController.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = KeyboardFixtureViewController.swift; sourceTree = "<group>"; };
		D00000000000000000000003 /* KeyboardHeightResolver.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = KeyboardHeightResolver.swift; sourceTree = "<group>"; };
		D00000000000000000000004 /* Info.plist */ = {isa = PBXFileReference; lastKnownFileType = text.plist.xml; path = Info.plist; sourceTree = "<group>"; };
		D00000000000000000000011 /* ExistingBehaviorTests.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = ExistingBehaviorTests.swift; sourceTree = "<group>"; };
		D00000000000000000000021 /* KeyboardTaskOracleTests.swift */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = KeyboardTaskOracleTests.swift; sourceTree = "<group>"; };
		D00000000000000000000030 /* UIKit.framework */ = {isa = PBXFileReference; lastKnownFileType = wrapper.framework; name = UIKit.framework; path = System/Library/Frameworks/UIKit.framework; sourceTree = SDKROOT; };
		D00000000000000000000031 /* XCTest.framework */ = {isa = PBXFileReference; lastKnownFileType = wrapper.framework; name = XCTest.framework; path = System/Library/Frameworks/XCTest.framework; sourceTree = SDKROOT; };
		D00000000000000000000040 /* KeyboardFixture.app */ = {isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = KeyboardFixture.app; sourceTree = BUILT_PRODUCTS_DIR; };
		D00000000000000000000041 /* KeyboardFixtureTests.xctest */ = {isa = PBXFileReference; explicitFileType = wrapper.cfbundle; includeInIndex = 0; path = KeyboardFixtureTests.xctest; sourceTree = BUILT_PRODUCTS_DIR; };
		D00000000000000000000042 /* KeyboardTaskOracleTests.xctest */ = {isa = PBXFileReference; explicitFileType = wrapper.cfbundle; includeInIndex = 0; path = KeyboardTaskOracleTests.xctest; sourceTree = BUILT_PRODUCTS_DIR; };
/* End PBXFileReference section */

/* Begin PBXFrameworksBuildPhase section */
		A10000000000000000000001 = {isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (B00000000000000000000004 /* UIKit.framework in Frameworks */,); runOnlyForDeploymentPostprocessing = 0; };
		A10000000000000000000002 = {isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (B00000000000000000000012 /* XCTest.framework in Frameworks */,); runOnlyForDeploymentPostprocessing = 0; };
		A10000000000000000000003 = {isa = PBXFrameworksBuildPhase; buildActionMask = 2147483647; files = (B00000000000000000000022 /* XCTest.framework in Frameworks */,); runOnlyForDeploymentPostprocessing = 0; };
/* End PBXFrameworksBuildPhase section */

/* Begin PBXGroup section */
		F00000000000000000000000 = {isa = PBXGroup; children = (F00000000000000000000001 /* KeyboardFixture */, F00000000000000000000002 /* KeyboardFixtureTests */, F00000000000000000000003 /* KeyboardTaskOracleTests */, F00000000000000000000005 /* Frameworks */, F00000000000000000000004 /* Products */,); sourceTree = "<group>"; };
		F00000000000000000000001 /* KeyboardFixture */ = {isa = PBXGroup; children = (D00000000000000000000001 /* AppDelegate.swift */, D00000000000000000000002 /* KeyboardFixtureViewController.swift */, D00000000000000000000003 /* KeyboardHeightResolver.swift */, D00000000000000000000004 /* Info.plist */,); path = KeyboardFixture; sourceTree = "<group>"; };
		F00000000000000000000002 /* KeyboardFixtureTests */ = {isa = PBXGroup; children = (D00000000000000000000011 /* ExistingBehaviorTests.swift */,); path = KeyboardFixtureTests; sourceTree = "<group>"; };
		F00000000000000000000003 /* KeyboardTaskOracleTests */ = {isa = PBXGroup; children = (D00000000000000000000021 /* KeyboardTaskOracleTests.swift */,); path = KeyboardTaskOracleTests; sourceTree = "<group>"; };
		F00000000000000000000004 /* Products */ = {isa = PBXGroup; children = (D00000000000000000000040 /* KeyboardFixture.app */, D00000000000000000000041 /* KeyboardFixtureTests.xctest */, D00000000000000000000042 /* KeyboardTaskOracleTests.xctest */,); name = Products; sourceTree = "<group>"; };
		F00000000000000000000005 /* Frameworks */ = {isa = PBXGroup; children = (D00000000000000000000030 /* UIKit.framework */, D00000000000000000000031 /* XCTest.framework */,); name = Frameworks; sourceTree = "<group>"; };
/* End PBXGroup section */

/* Begin PBXNativeTarget section */
		A00000000000000000000001 /* KeyboardFixture */ = {isa = PBXNativeTarget; buildConfigurationList = C00000000000000000000011; buildPhases = (A30000000000000000000001, A10000000000000000000001, A20000000000000000000001,); buildRules = (); dependencies = (); name = KeyboardFixture; productName = KeyboardFixture; productReference = D00000000000000000000040 /* KeyboardFixture.app */; productType = "com.apple.product-type.application"; };
		A00000000000000000000002 /* KeyboardFixtureTests */ = {isa = PBXNativeTarget; buildConfigurationList = C00000000000000000000012; buildPhases = (A30000000000000000000002, A10000000000000000000002, A20000000000000000000002,); buildRules = (); dependencies = (E10000000000000000000001,); name = KeyboardFixtureTests; productName = KeyboardFixtureTests; productReference = D00000000000000000000041 /* KeyboardFixtureTests.xctest */; productType = "com.apple.product-type.bundle.unit-test"; };
		A00000000000000000000003 /* KeyboardTaskOracleTests */ = {isa = PBXNativeTarget; buildConfigurationList = C00000000000000000000013; buildPhases = (A30000000000000000000003, A10000000000000000000003, A20000000000000000000003,); buildRules = (); dependencies = (E10000000000000000000002,); name = KeyboardTaskOracleTests; productName = KeyboardTaskOracleTests; productReference = D00000000000000000000042 /* KeyboardTaskOracleTests.xctest */; productType = "com.apple.product-type.bundle.unit-test"; };
/* End PBXNativeTarget section */

/* Begin PBXProject section */
		A00000000000000000000010 /* Project object */ = {isa = PBXProject; attributes = {BuildIndependentTargetsInParallel = 1; LastSwiftUpdateCheck = 2600; LastUpgradeCheck = 2600; TargetAttributes = {A00000000000000000000001 = {CreatedOnToolsVersion = 26.0;}; A00000000000000000000002 = {CreatedOnToolsVersion = 26.0; TestTargetID = A00000000000000000000001;}; A00000000000000000000003 = {CreatedOnToolsVersion = 26.0; TestTargetID = A00000000000000000000001;};};}; buildConfigurationList = C00000000000000000000010; compatibilityVersion = "Xcode 14.0"; developmentRegion = en; hasScannedForEncodings = 0; knownRegions = (en, Base,); mainGroup = F00000000000000000000000; productRefGroup = F00000000000000000000004 /* Products */; projectDirPath = ""; projectRoot = ""; targets = (A00000000000000000000001 /* KeyboardFixture */, A00000000000000000000002 /* KeyboardFixtureTests */, A00000000000000000000003 /* KeyboardTaskOracleTests */,); };
/* End PBXProject section */

/* Begin PBXResourcesBuildPhase section */
		A20000000000000000000001 = {isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
		A20000000000000000000002 = {isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
		A20000000000000000000003 = {isa = PBXResourcesBuildPhase; buildActionMask = 2147483647; files = (); runOnlyForDeploymentPostprocessing = 0; };
/* End PBXResourcesBuildPhase section */

/* Begin PBXSourcesBuildPhase section */
		A30000000000000000000001 = {isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = (B00000000000000000000001 /* AppDelegate.swift in Sources */, B00000000000000000000002 /* KeyboardFixtureViewController.swift in Sources */, B00000000000000000000003 /* KeyboardHeightResolver.swift in Sources */,); runOnlyForDeploymentPostprocessing = 0; };
		A30000000000000000000002 = {isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = (B00000000000000000000011 /* ExistingBehaviorTests.swift in Sources */,); runOnlyForDeploymentPostprocessing = 0; };
		A30000000000000000000003 = {isa = PBXSourcesBuildPhase; buildActionMask = 2147483647; files = (B00000000000000000000021 /* KeyboardTaskOracleTests.swift in Sources */,); runOnlyForDeploymentPostprocessing = 0; };
/* End PBXSourcesBuildPhase section */

/* Begin PBXTargetDependency section */
		E10000000000000000000001 = {isa = PBXTargetDependency; target = A00000000000000000000001 /* KeyboardFixture */; targetProxy = E00000000000000000000001; };
		E10000000000000000000002 = {isa = PBXTargetDependency; target = A00000000000000000000001 /* KeyboardFixture */; targetProxy = E00000000000000000000002; };
/* End PBXTargetDependency section */

/* Begin XCBuildConfiguration section */
		C10000000000000000000001 = {isa = XCBuildConfiguration; buildSettings = {ALWAYS_SEARCH_USER_PATHS = NO; CLANG_ENABLE_MODULES = YES; ENABLE_TESTABILITY = YES; GCC_C_LANGUAGE_STANDARD = gnu17; IPHONEOS_DEPLOYMENT_TARGET = 16.0; SDKROOT = iphoneos; SWIFT_VERSION = 5.0;}; name = Debug; };
		C10000000000000000000002 = {isa = XCBuildConfiguration; buildSettings = {ALWAYS_SEARCH_USER_PATHS = NO; CLANG_ENABLE_MODULES = YES; GCC_C_LANGUAGE_STANDARD = gnu17; IPHONEOS_DEPLOYMENT_TARGET = 16.0; SDKROOT = iphoneos; SWIFT_COMPILATION_MODE = wholemodule; SWIFT_VERSION = 5.0; VALIDATE_PRODUCT = YES;}; name = Release; };
		C10000000000000000000011 = {isa = XCBuildConfiguration; buildSettings = {CODE_SIGNING_ALLOWED = NO; CODE_SIGNING_REQUIRED = NO; GENERATE_INFOPLIST_FILE = NO; INFOPLIST_FILE = KeyboardFixture/Info.plist; PRODUCT_BUNDLE_IDENTIFIER = dev.xcodefixbench.keyboardfixture; PRODUCT_NAME = "$(TARGET_NAME)"; SUPPORTED_PLATFORMS = "iphonesimulator"; SUPPORTS_MACCATALYST = NO; SWIFT_EMIT_LOC_STRINGS = NO; TARGETED_DEVICE_FAMILY = 1;}; name = Debug; };
		C10000000000000000000012 = {isa = XCBuildConfiguration; buildSettings = {CODE_SIGNING_ALLOWED = NO; CODE_SIGNING_REQUIRED = NO; GENERATE_INFOPLIST_FILE = NO; INFOPLIST_FILE = KeyboardFixture/Info.plist; PRODUCT_BUNDLE_IDENTIFIER = dev.xcodefixbench.keyboardfixture; PRODUCT_NAME = "$(TARGET_NAME)"; SUPPORTED_PLATFORMS = "iphonesimulator"; SUPPORTS_MACCATALYST = NO; SWIFT_EMIT_LOC_STRINGS = NO; TARGETED_DEVICE_FAMILY = 1;}; name = Release; };
		C10000000000000000000021 = {isa = XCBuildConfiguration; buildSettings = {BUNDLE_LOADER = "$(TEST_HOST)"; CODE_SIGNING_ALLOWED = NO; CODE_SIGNING_REQUIRED = NO; GENERATE_INFOPLIST_FILE = YES; LD_RUNPATH_SEARCH_PATHS = "$(inherited) @executable_path/Frameworks @loader_path/Frameworks"; PRODUCT_BUNDLE_IDENTIFIER = dev.xcodefixbench.keyboardfixture.tests; PRODUCT_NAME = "$(TARGET_NAME)"; SUPPORTED_PLATFORMS = "iphonesimulator"; SWIFT_EMIT_LOC_STRINGS = NO; TEST_HOST = "$(BUILT_PRODUCTS_DIR)/KeyboardFixture.app/KeyboardFixture";}; name = Debug; };
		C10000000000000000000022 = {isa = XCBuildConfiguration; buildSettings = {BUNDLE_LOADER = "$(TEST_HOST)"; CODE_SIGNING_ALLOWED = NO; CODE_SIGNING_REQUIRED = NO; GENERATE_INFOPLIST_FILE = YES; LD_RUNPATH_SEARCH_PATHS = "$(inherited) @executable_path/Frameworks @loader_path/Frameworks"; PRODUCT_BUNDLE_IDENTIFIER = dev.xcodefixbench.keyboardfixture.tests; PRODUCT_NAME = "$(TARGET_NAME)"; SUPPORTED_PLATFORMS = "iphonesimulator"; SWIFT_EMIT_LOC_STRINGS = NO; TEST_HOST = "$(BUILT_PRODUCTS_DIR)/KeyboardFixture.app/KeyboardFixture";}; name = Release; };
		C10000000000000000000031 = {isa = XCBuildConfiguration; buildSettings = {BUNDLE_LOADER = "$(TEST_HOST)"; CODE_SIGNING_ALLOWED = NO; CODE_SIGNING_REQUIRED = NO; GENERATE_INFOPLIST_FILE = YES; LD_RUNPATH_SEARCH_PATHS = "$(inherited) @executable_path/Frameworks @loader_path/Frameworks"; PRODUCT_BUNDLE_IDENTIFIER = dev.xcodefixbench.keyboardfixture.oracle; PRODUCT_NAME = "$(TARGET_NAME)"; SUPPORTED_PLATFORMS = "iphonesimulator"; SWIFT_EMIT_LOC_STRINGS = NO; TEST_HOST = "$(BUILT_PRODUCTS_DIR)/KeyboardFixture.app/KeyboardFixture";}; name = Debug; };
		C10000000000000000000032 = {isa = XCBuildConfiguration; buildSettings = {BUNDLE_LOADER = "$(TEST_HOST)"; CODE_SIGNING_ALLOWED = NO; CODE_SIGNING_REQUIRED = NO; GENERATE_INFOPLIST_FILE = YES; LD_RUNPATH_SEARCH_PATHS = "$(inherited) @executable_path/Frameworks @loader_path/Frameworks"; PRODUCT_BUNDLE_IDENTIFIER = dev.xcodefixbench.keyboardfixture.oracle; PRODUCT_NAME = "$(TARGET_NAME)"; SUPPORTED_PLATFORMS = "iphonesimulator"; SWIFT_EMIT_LOC_STRINGS = NO; TEST_HOST = "$(BUILT_PRODUCTS_DIR)/KeyboardFixture.app/KeyboardFixture";}; name = Release; };
/* End XCBuildConfiguration section */

/* Begin XCConfigurationList section */
		C00000000000000000000010 = {isa = XCConfigurationList; buildConfigurations = (C10000000000000000000001, C10000000000000000000002,); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
		C00000000000000000000011 = {isa = XCConfigurationList; buildConfigurations = (C10000000000000000000011, C10000000000000000000012,); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
		C00000000000000000000012 = {isa = XCConfigurationList; buildConfigurations = (C10000000000000000000021, C10000000000000000000022,); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
		C00000000000000000000013 = {isa = XCConfigurationList; buildConfigurations = (C10000000000000000000031, C10000000000000000000032,); defaultConfigurationIsVisible = 0; defaultConfigurationName = Release; };
/* End XCConfigurationList section */
	};
	rootObject = A00000000000000000000010 /* Project object */;
}
"""

SCHEME = """<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion="2600" version="1.7">
  <BuildAction parallelizeBuildables="YES" buildImplicitDependencies="YES">
    <BuildActionEntries>
      <BuildActionEntry buildForTesting="YES" buildForRunning="YES" buildForProfiling="YES" buildForArchiving="YES" buildForAnalyzing="YES">
        <BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="A00000000000000000000001" BuildableName="KeyboardFixture.app" BlueprintName="KeyboardFixture" ReferencedContainer="container:KeyboardFixture.xcodeproj"/>
      </BuildActionEntry>
      <BuildActionEntry buildForTesting="YES" buildForRunning="NO" buildForProfiling="NO" buildForArchiving="NO" buildForAnalyzing="NO">
        <BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="A00000000000000000000002" BuildableName="KeyboardFixtureTests.xctest" BlueprintName="KeyboardFixtureTests" ReferencedContainer="container:KeyboardFixture.xcodeproj"/>
      </BuildActionEntry>
      <BuildActionEntry buildForTesting="YES" buildForRunning="NO" buildForProfiling="NO" buildForArchiving="NO" buildForAnalyzing="NO">
        <BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="A00000000000000000000003" BuildableName="KeyboardTaskOracleTests.xctest" BlueprintName="KeyboardTaskOracleTests" ReferencedContainer="container:KeyboardFixture.xcodeproj"/>
      </BuildActionEntry>
    </BuildActionEntries>
  </BuildAction>
  <TestAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.DebuggerFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv="YES">
    <Testables>
      <TestableReference skipped="NO"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="A00000000000000000000002" BuildableName="KeyboardFixtureTests.xctest" BlueprintName="KeyboardFixtureTests" ReferencedContainer="container:KeyboardFixture.xcodeproj"/></TestableReference>
      <TestableReference skipped="NO"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="A00000000000000000000003" BuildableName="KeyboardTaskOracleTests.xctest" BlueprintName="KeyboardTaskOracleTests" ReferencedContainer="container:KeyboardFixture.xcodeproj"/></TestableReference>
    </Testables>
  </TestAction>
  <LaunchAction buildConfiguration="Debug" selectedDebuggerIdentifier="Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier="Xcode.DebuggerFoundation.Launcher.LLDB" launchStyle="0" useCustomWorkingDirectory="NO" ignoresPersistentStateOnLaunch="NO" debugDocumentVersioning="YES" debugServiceExtension="internal" allowLocationSimulation="YES">
    <BuildableProductRunnable runnableDebuggingMode="0"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="A00000000000000000000001" BuildableName="KeyboardFixture.app" BlueprintName="KeyboardFixture" ReferencedContainer="container:KeyboardFixture.xcodeproj"/></BuildableProductRunnable>
  </LaunchAction>
  <ProfileAction buildConfiguration="Release" shouldUseLaunchSchemeArgsEnv="YES" savedToolIdentifier="" useCustomWorkingDirectory="NO" debugDocumentVersioning="YES"><BuildableProductRunnable runnableDebuggingMode="0"><BuildableReference BuildableIdentifier="primary" BlueprintIdentifier="A00000000000000000000001" BuildableName="KeyboardFixture.app" BlueprintName="KeyboardFixture" ReferencedContainer="container:KeyboardFixture.xcodeproj"/></BuildableProductRunnable></ProfileAction>
  <AnalyzeAction buildConfiguration="Debug"/>
  <ArchiveAction buildConfiguration="Release" revealArchiveInOrganizer="YES"/>
</Scheme>
"""


def main() -> None:
    scheme_dir = PROJECT / "xcshareddata" / "xcschemes"
    scheme_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT / "project.pbxproj").write_text(PBXPROJ, encoding="utf-8")
    (scheme_dir / "KeyboardFixture.xcscheme").write_text(SCHEME, encoding="utf-8")
    print(f"generated {PROJECT}")


if __name__ == "__main__":
    main()
