#!/usr/bin/env python3
"""Generate trigger-level VectorRAG patient cases from input_nodes.json.

The generated labels intentionally distinguish between:

* rows *triggered* by at least one mentioned input node; and
* rows whose complete rule condition is satisfied (left null for later review).

The first 70 cases partition the clinically usable input nodes exactly once.
Administrative, workflow, availability, and otherwise non-patient-like nodes
are excluded with an explicit audit trail. The final 30 cases reuse complete
base cases to test lexical robustness (20 terminology synonyms and 10 freer
paraphrases).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "trigger-gold-standard-v2"
TRIGGER_SEMANTICS = (
    "A row is expected here when at least one intended input node occurs in that "
    "row. This is not a claim that every condition of the row is fulfilled."
)


EXCLUDED_NODES: dict[str, str] = {
    "input_c0747fbf391ef3673817": "Non-specific image-quality workflow condition.",
    "input_63fccf01a6f58c6c13f2": "Diagnostic-test feasibility rather than a patient finding.",
    "input_7aae9a0105bdb9c0d69a": "Diagnostic-test availability rather than a patient finding.",
    "input_1a5a78bbde0b8d8ee9ec": "Diagnostic-test availability rather than a patient finding.",
    "input_48a7dd925ddc6243dd86": "Procedure timing rather than a patient finding or history.",
    "input_0eceabe3a122de90a6f0": "Prognostic eligibility wording rather than a concrete patient finding.",
    "input_55d5655678ce98976009": "Prognostic eligibility wording rather than a concrete patient finding.",
    "input_d1eb9bd7bbb6c616227e": "Non-diagnostic test workflow state.",
    "input_f66c3f65291449e65fdb": "Diagnostic-test availability rather than a patient finding.",
    "input_d9014b9ae2fb40a58d45": "Unknown risk status is not a positive clinical finding.",
    "input_4c9133840c4bafc3e689": "Test-cooperation workflow constraint.",
    "input_8efa9c8c812ac90268c3": "Inconclusive test workflow state.",
    "input_6a0df43c45a56594b2ef": "Investigation indication rather than a patient finding.",
    "input_3108c6ee3c4ffcec1d6d": "Diagnostic-test availability rather than a patient finding.",
    "input_ff9feb22fd12788403c3": "Uncertain test conclusion rather than a concrete patient finding.",
    "input_53e87a34be7accfc61e5": "Treatment-suitability assessment rather than a patient finding.",
    "input_18aa979a52cca1325ebc": "Procedure-end workflow state.",
    "input_b81440569307a1d63665": "Assessment workflow label rather than a patient finding.",
    "input_c82bd9b26154fda7ee1b": "Generic procedure label with no clinical specificity.",
    "input_65620d725081778af9ea": "Scheduling state rather than procedure history.",
    "input_99df01f3dede3bd6ee76": "Imaging-selection workflow state.",
    "input_0a7ba79e1b5ce9f0d80c": "Heart Team proposal rather than a patient finding.",
    "input_baac7c83f128211e71db": "Generic procedure label with no timing or clinical specificity.",
    "input_f1cc271f2766d04fd911": "Treatment-decision workflow state.",
    "input_0c6da199d8a1ecb775ee": "Imaging-selection workflow state.",
    "input_58f17979dd2faea7c755": "Scheduling state rather than procedure history.",
    "input_80b39749666f8a735e38": "Multidisciplinary-team qualifier rather than a patient finding.",
    "input_36ee3cfb054025bd0b0a": "Vague case-complexity label tied to a treatment decision.",
}


# Each lexical variant replaces one mention in an existing 3- or 4-node base
# case. The remaining mentions and all labels are inherited unchanged.
SNOMED_VARIANTS: list[dict[str, Any]] = [
    {
        "target_node_id": "input_6c180966d76985e738fa",
        "replacement": "AF",
        "concept_id": "49436004",
        "preferred_term": "Atrial fibrillation",
        "synonym_used": "AF",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/49436004",
    },
    {
        "target_node_id": "input_5812d323f9927cd41746",
        "replacement": "cardiac failure",
        "concept_id": "84114007",
        "preferred_term": "Heart failure",
        "synonym_used": "Cardiac failure",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/84114007",
    },
    {
        "target_node_id": "input_38457cb950b5869427f7",
        "replacement": "cardiac insufficiency",
        "concept_id": "84114007",
        "preferred_term": "Heart failure",
        "synonym_used": "Cardiac insufficiency",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/84114007",
    },
    {
        "target_node_id": "input_5812d323f9927cd41746",
        "replacement": "myocardial failure",
        "concept_id": "84114007",
        "preferred_term": "Heart failure",
        "synonym_used": "Myocardial failure",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/84114007",
    },
    {
        "target_node_id": "input_d7012b06e035bf6de0a6",
        "replacement": "high blood pressure",
        "concept_id": "38341003",
        "preferred_term": "Hypertensive disorder",
        "synonym_used": "High blood pressure",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/38341003",
    },
    {
        "target_node_id": "input_d7012b06e035bf6de0a6",
        "replacement": "systemic arterial hypertension",
        "concept_id": "38341003",
        "preferred_term": "Hypertensive disorder",
        "synonym_used": "Systemic arterial hypertension",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/38341003",
    },
    {
        "target_node_id": "input_20ef046fa732f2290278",
        "replacement": "coronary heart disease",
        "concept_id": "53741008",
        "preferred_term": "Coronary arteriosclerosis",
        "synonym_used": "Coronary heart disease",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/53741008",
    },
    {
        "target_node_id": "input_22a3a2ecad0b96fcf4e4",
        "replacement": "arteriosclerotic heart disease",
        "concept_id": "53741008",
        "preferred_term": "Coronary arteriosclerosis",
        "synonym_used": "Arteriosclerotic heart disease",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/53741008",
    },
    {
        "target_node_id": "input_20ef046fa732f2290278",
        "replacement": "coronary sclerosis",
        "concept_id": "53741008",
        "preferred_term": "Coronary arteriosclerosis",
        "synonym_used": "Coronary sclerosis",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/53741008",
    },
    {
        "target_node_id": "input_ebc0d3a25177a5391c8b",
        "replacement": "type II diabetes mellitus",
        "concept_id": "44054006",
        "preferred_term": "Type 2 diabetes mellitus",
        "synonym_used": "Type II diabetes mellitus",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/44054006",
    },
    {
        "target_node_id": "input_95d02c5292c000019484",
        "replacement": "DM",
        "concept_id": "73211009",
        "preferred_term": "Diabetes mellitus",
        "synonym_used": "DM",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/73211009",
    },
    {
        "target_node_id": "input_133b7c7384d1ea058225",
        "replacement": "cardiac angina",
        "concept_id": "194828000",
        "preferred_term": "Angina pectoris",
        "synonym_used": "Cardiac angina",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/194828000",
    },
    {
        "target_node_id": "input_833f9371ce973fce7b19",
        "replacement": "frequent anginal syndrome symptoms",
        "concept_id": "194828000",
        "preferred_term": "Angina pectoris",
        "synonym_used": "Anginal syndrome",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/194828000",
    },
    {
        "target_node_id": "input_90e908eb2d1f46b346a5",
        "replacement": "stenocardia",
        "concept_id": "194828000",
        "preferred_term": "Angina pectoris",
        "synonym_used": "Stenocardia",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/194828000",
    },
    {
        "target_node_id": "input_d8790a0001d4de9c3781",
        "replacement": "angina of effort during light activity",
        "concept_id": "300995000",
        "preferred_term": "Exercise-induced angina",
        "synonym_used": "Angina of effort",
        "verification_url": "https://www.ncbi.nlm.nih.gov/medgen/661193",
    },
    {
        "target_node_id": "input_d8790a0001d4de9c3781",
        "replacement": "exertional angina during light activity",
        "concept_id": "300995000",
        "preferred_term": "Exercise-induced angina",
        "synonym_used": "Exertional angina",
        "verification_url": "https://www.ncbi.nlm.nih.gov/medgen/661193",
    },
    {
        "target_node_id": "input_483b9c3780af0ca2f06f",
        "replacement": "isolated variant angina",
        "concept_id": "87343002",
        "preferred_term": "Prinzmetal angina",
        "synonym_used": "Variant angina",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/87343002",
    },
    {
        "target_node_id": "input_71e5ae1088f7d1a2954d",
        "replacement": "breathlessness on exertion",
        "concept_id": "60845006",
        "preferred_term": "Dyspnea on exertion",
        "synonym_used": "Breathlessness on exertion",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/60845006",
    },
    {
        "target_node_id": "input_bcc91a4863046b079361",
        "replacement": "heart failure with preserved ejection fraction",
        "concept_id": "446221000",
        "preferred_term": "Heart failure with normal ejection fraction",
        "synonym_used": "Heart failure with preserved ejection fraction",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/446221000",
    },
    {
        "target_node_id": "input_9f720232c0ac73e59292",
        "replacement": "LBBB",
        "concept_id": "63467002",
        "preferred_term": "Left bundle branch block",
        "synonym_used": "LBBB",
        "verification_url": "https://purl.bioontology.org/ontology/SNOMEDCT/63467002",
    },
]


FREE_VARIANTS: list[dict[str, str]] = [
    {
        "target_node_id": "input_6c180966d76985e738fa",
        "replacement": "an irregular atrial rhythm",
    },
    {
        "target_node_id": "input_d8790a0001d4de9c3781",
        "replacement": "chest tightness during light exertion",
    },
    {
        "target_node_id": "input_71e5ae1088f7d1a2954d",
        "replacement": "shortness of breath during ordinary activity",
    },
    {
        "target_node_id": "input_c43f500eb3d33fe2f73b",
        "replacement": "coronary disease affecting several vessels",
    },
    {
        "target_node_id": "input_5460024a81a4619e9f18",
        "replacement": "weak left ventricular pumping due to systolic heart failure",
    },
    {
        "target_node_id": "input_4708c8c1066e3917f2fa",
        "replacement": "a history of open-heart bypass surgery",
    },
    {
        "target_node_id": "input_7fd81195828ceda76c88",
        "replacement": "a previous catheter-based coronary stent procedure",
    },
    {
        "target_node_id": "input_f68fb8e5f31e3aa9a5bc",
        "replacement": "an inability to tolerate beta-blocking medication",
    },
    {
        "target_node_id": "input_c438977c89c3fb75b1f5",
        "replacement": "a marked tendency to bleed",
    },
    {
        "target_node_id": "input_0963703efc35f4087ed3",
        "replacement": "advanced kidney failure",
    },
]


CLAUSE_OVERRIDES = {
    "input_3ee8db337f48d94162bb": "persistent angina or ischaemia symptoms",
    "input_3c6a25c3e58053de2d13": "high surgical risk",
    "input_c4ef80b61da260e8487e": "persistent symptoms despite medical treatment",
    "input_a14e5e921b349937c753": "no clinical heart failure",
    "input_26e77f9b7685423eaf0a": "no contraindication to combined beta-blocker and DHP-CCB therapy",
    "input_4a2a42262a40c3015ba4": "no myocardial infarction in the prior 40 days",
    "input_5af232fa662145ad0d34": "no reversible cause of the ventricular arrhythmia",
    "input_791160d7a235b006ed01": "no indication for oral anticoagulation",
    "input_339a073dac8ee026c13d": "no previous coronary revascularization",
    "input_6dbf4d0ee68f9f05dc6d": "an LDL-C reduction of only 35% from baseline",
    "input_797bf330c5a46789dd2b": "an LVEF of 32%",
    "input_6c2ab456c39505ea45a6": "an LVEF of 33% despite guideline-directed therapy",
    "input_f8956f1a23d56a23c04c": "an LVEF of 30% despite three months of optimized therapy",
    "input_45194e4ad3cf31d09308": "an LVEF of 36%",
    "input_48c408b0d324656ea19c": "an LVEF of 37%",
    "input_df5467d7dafc528f182d": "an LVEF of 45%",
    "input_33398bb7dd57c11caae7": "an LVEF of 30%",
    "input_3fdaac319ac4883df3e5": "a preserved LVEF of 55%",
    "input_0eed264b3080baf18e5b": "a pre-test likelihood of obstructive CAD of 10%",
    "input_0fa581a4b52756f340e4": "a pre-test likelihood of obstructive CAD of 12%",
    "input_a374595777d4894bb5b3": "a pre-test likelihood of obstructive CAD of 35%",
    "input_5dad92d638f29af37bf0": "a pre-test likelihood of obstructive CAD of 65%",
    "input_aea6f7ed694005f26197": "a pre-test likelihood of obstructive CAD of 25%",
    "input_09fe9651b4795169480f": "a pre-test likelihood of obstructive CAD of 90%",
    "input_66d0161346b2485a9c46": "a pre-test likelihood of obstructive CAD of 3%",
    "input_8d5a55bb696f49cfbe77": "a pre-test likelihood of obstructive CAD of 40%",
    "input_aeeb0b722899f27e60f0": "a clinical likelihood of obstructive CAD of 90%",
    "input_9ebd6756f60a54102c21": "an age of 79 years",
    "input_a52c8a8aa1490d14f1a0": "an eGFR of 24 mL/min/1.73 m²",
    "input_78ce98f8817d2711bf86": "an LDL-C of 70 mg/dL despite lipid-lowering therapy",
    "input_b67d15533d99854d0c27": "a QRS duration of 160 ms",
    "input_d2392b1ea7ed65434db1": "0.2 mV ST-segment depression on the resting ECG",
    "input_bb92f1f8005d4a937b0c": "a SYNTAX score of 28",
    "input_a84377c1f1408783713a": "a SYNTAX score of 18",
    "input_249210d3c9d81644ff1f": "a BMI of 29 kg/m²",
    "input_004d40b9acf8c220a4ad": "overweight with a BMI of 29 kg/m²",
    "input_e52167297dd90aa645a2": "prior myocardial infarction",
    "input_4708c8c1066e3917f2fa": "a history of CABG",
    "input_7fd81195828ceda76c88": "a history of PCI",
    "input_91212cbc0b2cf6270008": "a history of uncomplicated PCI",
    "input_115cd414f5533a9282be": "remote history of PCI",
    "input_85fcc8d3e4d1e491cd37": "a history of PCI",
    "input_f1cc271f2766d04fd911": "a decision being made about revascularization and its modality",
    "input_61400190e4cf89ed494b": "a history of high-risk PCI for complex CAD",
    "input_99df01f3dede3bd6ee76": "planned PET myocardial perfusion imaging",
    "input_0c6da199d8a1ecb775ee": "planned SPECT myocardial perfusion imaging",
    "input_9f7c2e42420c70c68b18": "initial treatment with a beta-blocker",
    "input_1a8d52cceb216a7963cc": "initial treatment with a calcium-channel blocker",
    "input_2e5566397b298c8061cc": "current beta-blocker treatment",
    "input_9fb45801dec5bd68c8b0": "current calcium-channel blocker treatment",
    "input_ab04f1704fc0d231f052": "inadequate symptom control on beta-blocker therapy",
    "input_e91c896a08568aef9594": "inadequate symptom control on calcium-channel blocker therapy",
    "input_21e61db182efee63b612": "maximum tolerated statin therapy",
    "input_7ddbe5bcd374e8cfeddb": "maximum tolerated statin therapy with ezetimibe",
    "input_c151bff5c20a5d88496d": "initial ticagrelor-based dual antiplatelet therapy",
    "input_b010388cd9bb477c5f49": "an indication for vitamin K antagonist therapy",
    "input_48e1da2a1a7fe6a7e42a": "current antithrombotic therapy",
    "input_7144fde663654b295eb5": "current long-acting nitrate therapy",
    "input_e6f7ad4eef663fd4e2ce": "concurrent phosphodiesterase inhibitor therapy",
    "input_f5d1e0a16db123031076": "angina not controlled by initial beta-blocker or CCB monotherapy",
    "input_f68fb8e5f31e3aa9a5bc": "intolerance or contraindication to beta-blockers",
    "input_fc1dc61d44568da6d9d3": "intolerance or contraindication to calcium-channel blockers",
    "input_fd8c7673e252079d821a": "inadequate symptom control on beta-blockers and/or calcium-channel blockers",
    "input_bdefa465363787063e6f": "inadequately controlled anginal symptoms",
    "input_6c180966d76985e738fa": "atrial fibrillation",
    "input_8ef32e47ef8e16b11186": "high-degree AV block requiring ventricular pacing",
    "input_d1811663333a23d2759e": "a history of acute coronary syndrome with subsequent stabilization",
    "input_61a97c0d58a89d9ac4d1": "concern about stent-thrombosis risk",
    "input_72fb283a2abde3b77e2c": "concern about ischaemic-stroke risk",
    "input_dedf853a6462dc6e6cb2": "high bleeding risk outweighing stent-thrombosis or ischaemic-stroke concerns",
    "input_ef83ae3cbc1f4c1c6ee3": "maximally tolerated statin therapy",
    "input_3e5a6f8343bb2412bc2e": "multivessel obstructive CAD",
    "input_d6f5dcc8681e96d41f0d": "symptoms suspected to be of myocardial ischaemic origin",
    "input_1c2ef533e38d7ad9d619": "recovery from a haemodynamically unstable ventricular arrhythmia",
    "input_30f016c4bfb8291cfcfe": "multivessel disease",
    "input_345db7291ed05ec25c63": "complex coronary disease under evaluation for revascularization",
    "input_a587c946fad3ca85f047": "complex CAD",
    "input_fce849fdc17583f45672": "persistent angina",
    "input_67fe43c4195ba8a7ac52": "symptoms despite guideline-directed medical treatment",
    "input_66fd48b3167c845f23d5": "low surgical risk",
    "input_6e0bbfffcf685de38507": "significant left main coronary stenosis",
    "input_004fe4f69751b496927f": "significant left main stenosis with a SYNTAX score of 18",
    "input_0b845e74f1fc6ffb3879": "current digitalis treatment",
    "input_3108c6ee3c4ffcec1d6d": "unavailable non-invasive imaging tests",
    "input_1a5a78bbde0b8d8ee9ec": "available CCTA",
    "input_f66c3f65291449e65fdb": "available functional imaging tests",
    "input_22f99f4831fd01e90b44": "coronary artery calcification visible on a previous chest CT",
    "input_63fccf01a6f58c6c13f2": "CACS not feasible for detecting atherosclerotic disease",
    "input_7aae9a0105bdb9c0d69a": "unavailable CACS",
    "input_b4188201fd112218a171": "inoperable coronary disease",
    "input_86be9bdfb292c19ecf1b": "fitness for cardiac surgery",
    "input_206bbd2673db2c65fadc": "current symptoms",
    "input_24f2b639a26469346be9": "high risk of adverse events",
    "input_cc20ae0da24c8e9b7191": "suspected chronic coronary syndrome",
    "input_d9014b9ae2fb40a58d45": "unknown high-risk status for adverse events",
    "input_0eceabe3a122de90a6f0": "expected survival >1 year with good functional status",
    "input_55d5655678ce98976009": "expected survival substantially >1 year with good functional status",
    "input_0a7ba79e1b5ce9f0d80c": "a Heart Team treatment proposal",
    "input_80b39749666f8a735e38": "review by the Heart Team",
    "input_d8da1af39c38ae05d127": "chronic coronary syndrome",
    "input_6a0df43c45a56594b2ef": "an indication for invasive coronary angiography",
    "input_48a7dd925ddc6243dd86": "assessment during invasive coronary angiography",
    "input_58f17979dd2faea7c755": "planned surgical revascularization",
    "input_65620d725081778af9ea": "planned percutaneous revascularization",
    "input_18aa979a52cca1325ebc": "completion of myocardial revascularization",
    "input_02693eb28e6c037cd180": "suspected chronic coronary syndrome",
    "input_5dc19fd90472f09c3069": "complex coronary anatomy with thrombotic risk outweighing bleeding risk",
    "input_8e6546268a7282d3ef65": "anatomically complex coronary lesions",
    "input_833f9371ce973fce7b19": "frequent angina",
    "input_90e908eb2d1f46b346a5": "angina with myocardial ischaemia",
    "input_d8790a0001d4de9c3781": "angina during light exercise",
    "input_46db42052a49a3e0f1c4": "persistent anginal-equivalent symptoms",
    "input_421b1d8bfe8a4b671fb9": "suspected ANOCA",
    "input_469da2dd74cc40e6266b": "ANOCA",
    "input_8074cbf01a76d2b3ea55": "suspected INOCA",
    "input_5ab5fd7f60307e05613d": "suspected INOCA",
    "input_95412f99a8f53fbf5f24": "chronic coronary syndrome and a history of CABG",
    "input_205697291d49fbc8f694": "failure of a previous CABG graft",
    "input_20ef046fa732f2290278": "coronary artery disease",
    "input_d8c85e925324a8a04cd5": "chronic coronary syndrome",
    "input_410e47184bb8669414f4": "chronic coronary syndrome",
    "input_1461e26062c41f8d36fe": "chest pain during an episode suggestive of myocardial ischaemia",
    "input_2684d4767adea4a09ed2": "chest pain triggered by emotional stress",
    "input_4033697d6437ceac5472": "suspected chronic coronary syndrome",
    "input_82c7164b5eaa057dd601": "symptomatic chronic coronary syndrome",
    "input_70ea764d1f5aa7836877": "symptomatic chronic coronary syndrome",
    "input_39de6f7bb974e37bd353": "complex coronary artery disease",
    "input_e686eb8acff7f184e108": "complex significant double-vessel disease involving the proximal LAD",
    "input_0e9acdaac9484ad3c64f": "complex significant single-vessel disease involving the proximal LAD",
    "input_c5730e4ee9320c04515d": "coronary disease of uncertain functional significance on CCTA",
    "input_1726e54770bd8bb2a9d4": "severely limiting angina",
    "input_49d4d6beeb00dad1c714": "progressive left ventricular systolic dysfunction without a reversible cause",
    "input_6414f73ddc2997da8dfc": "increased ischaemic risk",
    "input_41919476b5361287976d": "functionally significant left main stenosis",
    "input_71d28ef743afeaaa3554": "functionally significant multivessel coronary disease",
    "input_2758f4aa977da43943a3": "functionally significant single-vessel disease involving the proximal LAD",
    "input_f2b00052f29b737034c1": "functionally significant three-vessel disease",
    "input_4f9d2a0af4d781aa7410": "functionally significant two-vessel disease involving the proximal LAD",
    "input_38457cb950b5869427f7": "heart failure",
    "input_5812d323f9927cd41746": "heart failure",
    "input_f3f1acf7d48b320f02b1": "heart failure with mildly reduced ejection fraction",
    "input_61d9fea7c9eda9f1643d": "heart failure with preserved ejection fraction",
    "input_5460024a81a4619e9f18": "heart failure with reduced ejection fraction",
    "input_55739003e983a5362060": "heart failure with reduced ejection fraction",
    "input_9eb9a70a3b76db41ddee": "symptomatic heart failure with reduced ejection fraction",
    "input_bcc91a4863046b079361": "heart failure with normal ejection fraction",
    "input_55baaec8ac4ce1038feb": "high bleeding risk",
    "input_19e86df1b068ed3e91f5": "high ischaemic risk outweighing bleeding risk",
    "input_5596443eead276467caf": "persistent high ischaemic risk",
    "input_f7046d6ccf71e7d25860": "drug-eluting coronary stent restenosis",
    "input_ba7e59dcd80f54a7c5ad": "an indication for long-term oral anticoagulation",
    "input_fe9219fc7a8a1f681f5d": "a long-term indication for oral anticoagulation",
    "input_470f2ef2362f9ee37026": "intermediate stenosis in a proximal coronary segment on CCTA",
    "input_967ff4bfd18d0036afec": "left bundle branch block QRS morphology",
    "input_e6b011e95b9a809fef9f": "left main coronary artery disease",
    "input_8ef1d8a9f970f4313096": "left main coronary lesions",
    "input_a0cbe969d8c5b321ce7f": "left ventricular systolic dysfunction",
    "input_89d637c1ab28b6888ab4": "long coronary lesions",
    "input_26eb741df0a8358f3886": "low surgical risk",
    "input_10fc38345c2c94843ad6": "microvascular angina",
    "input_24d74913be20e99f03b7": "microvascular angina with reduced coronary flow reserve",
    "input_209ea8f3cbee2dfaa6c3": "non-obstructive epicardial coronary arteries",
    "input_6c30e666810b81bf4e89": "normal epicardial coronary arteries",
    "input_f35408d5af3f54b36c5c": "suspected obstructive coronary artery disease",
    "input_19cc73c60d606f5d3b0e": "obstructive coronary disease refractory to medical therapy and revascularization",
    "input_3cdb97cd3eca133813ba": "overlapping coronary vasomotor endotypes",
    "input_3e781b2e094a59a4c3a2": "a history of CABG without prior LIMA use",
    "input_0ad88fdbf0857e1b3e32": "a history of CABG without a patent LIMA graft to the LAD",
    "input_1e7e0d784abffaa3604f": "persistent angina or anginal-equivalent symptoms",
    "input_a9a93c6e7d3ebd19d970": "recurrent angina with impaired quality of life",
    "input_f6092c0d82c07b1f5dfc": "recurrent myocardial ischaemia with impaired quality of life",
    "input_835181bba12394e119ef": "recurrent rest angina",
    "input_c993bf8f6e686a8a29e6": "a recurrent atherothrombotic event",
    "input_0963703efc35f4087ed3": "severe renal failure",
    "input_24ec61564b3b700cad44": "severe symptoms despite guideline-directed therapy",
    "input_af025d9f3945e4715c1e": "significant double-vessel disease involving the proximal LAD",
    "input_d12108a6ebc2bb41a04e": "significant double-vessel disease not involving the proximal LAD",
    "input_4a8a35460e318e8f4f78": "significant double-vessel disease not involving the proximal LAD and unsuitable for PCI",
    "input_9590092833635227e9e7": "significant left main stenosis of intermediate anatomical complexity",
    "input_3408c69e814fc1067c75": "significant multivessel coronary artery disease",
    "input_c3b6bc08345279deb8bc": "significant single-vessel disease involving the proximal LAD",
    "input_8a6b260fc0942bd365d7": "significant single-vessel disease not involving the proximal LAD",
    "input_9424265bf8ed9730036c": "significant single-vessel disease not involving the proximal LAD and unsuitable for PCI",
    "input_3ad181fe4c2d2597ab1d": "significant three-vessel disease of low-to-intermediate complexity",
    "input_8c411b9df1d888eb4943": "signs and symptoms suggestive of heart failure",
    "input_c6db0e59308070156137": "rest angina with transient ST-segment changes relieved by nitrates",
    "input_99a2422cd1a0b4e74c49": "suspected acute coronary syndrome based on symptoms and ECG findings",
    "input_5c3a33db475e01ae91c6": "suspected ANOCA with anginal symptoms and non-obstructive arteries",
    "input_8b8132aef3411bb25057": "suspected chronic coronary syndrome requiring diagnostic clarification",
    "input_a85092f0ebe75fd4a6f5": "new symptoms strongly suggestive of obstructive coronary disease during light exercise",
    "input_429f9acc6be750eb875f": "no ventricular arrhythmia during the first 48 hours after myocardial infarction",
    "input_2bca70f4b61ba02bb0ad": "severe obesity with a BMI of 42 kg/m²",
    "input_dfe24d865d9990b9df17": "obesity",
    "input_58d4a0d1b25805e36d1f": "statin intolerance",
    "input_e35a22a091e1ca1ca67d": "no current symptoms",
    "input_163821185e2840f66fe4": "symptomatic heart failure",
    "input_ab44c2813f3a650814ed": "symptomatic heart failure of ischaemic origin",
    "input_b9fbd4e844daa446ebb5": "a symptomatic presentation",
    "input_1d32d76675af980b3b05": "high cardiovascular event risk",
    "input_77935ee3063f45289d6a": "intolerance to ARNI therapy",
    "input_9aaa30efe2c9c4f87c42": "intolerance to ACE inhibitor therapy",
    "input_5d120d5cfd120f076d56": "current single or dual antiplatelet therapy",
    "input_d36f8084f867fc882390": "current ezetimibe therapy",
    "input_4ffa3599fd1a7a45b188": "NYHA class II symptoms",
    "input_23286f471e69305498da": "NYHA class III symptoms",
    "input_72063e466ab889fb3158": "female sex",
    "input_bf26dfe434c0c2541fbb": "a history of CABG",
    "input_ca18c337665f28616217": "a history of PCI",
    "input_0871057dc2aeb90cb92b": "a history of high-thrombotic-risk coronary stenting",
    "input_11d935c3928853b20c0b": "true coronary bifurcation lesions",
    "input_2e082343bc71616adea8": "increased risk of gastrointestinal bleeding",
    "input_9626ee3a4a8457a3bd89": "maximum tolerated statin therapy with ezetimibe",
}


MANUAL_GROUP_SWAPS: list[tuple[str, str]] = [
    # Preserve a coherent risk balance in both antithrombotic cases.
    ("input_19e86df1b068ed3e91f5", "input_6414f73ddc2997da8dfc"),
    # Keep ischaemic heart failure with established CAD rather than normal arteries.
    ("input_ab44c2813f3a650814ed", "input_55739003e983a5362060"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def row_id(occurrence: dict[str, Any]) -> str:
    return (
        f"{occurrence['file']}::table_{occurrence['table_id']}::"
        f"row_{occurrence['row_index_zero_based']}"
    )


NORMALIZED_ORIGINAL_CACHE: dict[str, str] = {}
NORMALIZED_CLAUSE_CACHE: dict[str, str] = {}
SNOMED_ID_CACHE: dict[str, frozenset[str]] = {}
MUTEX_CACHE: dict[str, dict[str, str]] = {}
COMPATIBILITY_CACHE: dict[tuple[str, str], bool] = {}


def normalized_original(node: dict[str, Any]) -> str:
    node_id = node["node_id"]
    if node_id in NORMALIZED_ORIGINAL_CACHE:
        return NORMALIZED_ORIGINAL_CACHE[node_id]
    text = node["entity_original"].casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    result = re.sub(r"\s+", " ", text).strip()
    NORMALIZED_ORIGINAL_CACHE[node_id] = result
    return result


def normalized_clause(node: dict[str, Any]) -> str:
    node_id = node["node_id"]
    if node_id in NORMALIZED_CLAUSE_CACHE:
        return NORMALIZED_CLAUSE_CACHE[node_id]
    result = re.sub(r"[^a-z0-9]+", " ", clause_for_node(node).casefold())
    result = re.sub(r"\s+", " ", result).strip()
    NORMALIZED_CLAUSE_CACHE[node_id] = result
    return result


def snomed_ids(node: dict[str, Any]) -> frozenset[str]:
    node_id = node["node_id"]
    if node_id in SNOMED_ID_CACHE:
        return SNOMED_ID_CACHE[node_id]
    result = frozenset({
        str(item.get("snomed_id", "")).strip()
        for item in node.get("entity_standardized_list", [])
        if str(item.get("snomed_id", "")).strip()
    })
    SNOMED_ID_CACHE[node_id] = result
    return result


def text_blob(node: dict[str, Any]) -> str:
    return f"{node['entity_original']} {node.get('context', '')}".casefold()


def category_tags(node: dict[str, Any]) -> set[str]:
    text = text_blob(node)
    tags: set[str] = {node.get("role", "unknown").casefold()}
    patterns = {
        "angina": r"angin|chest pain|ischaemi|ischemi|jaw|upper back|neck pain|arms",
        "cad_anatomy": r"coronary|vessel|lesion|lad|left main|stenos|syntax|calcif",
        "heart_failure": r"heart failure|\bhf(?:ref|pef|mref)?\b|lvef|ventricular|nyha|congestion",
        "rhythm": r"fibrillation|arrhythm|sinus rhythm|bundle branch|qrs|av block|tachy",
        "bleeding_thrombosis": r"bleed|thrombo|antiplatelet|anticoag|oac|ticagrelor|vitamin k",
        "revascularization": r"pci|cabg|revascular|stent|heart team|intervention",
        "diagnostics": r"ccta|imaging|echocardi|pet|spect|ica|pre-test|clinical likelihood|ecg",
        "lipids": r"ldl|statin|ezetimibe",
        "antianginal": r"beta-block|ccb|nitrate|digitalis|phosphodiesterase",
        "comorbidity": r"diabet|hypertension|renal|egfr|hiv|obes|overweight|bmi|cardiomyopathy",
        "risk": r"risk|operable|surgically|surviv|adverse event",
    }
    for tag, pattern in patterns.items():
        if re.search(pattern, text):
            tags.add(tag)
    return tags


def mutex_labels(node: dict[str, Any]) -> dict[str, str]:
    node_id = node["node_id"]
    if node_id in MUTEX_CACHE:
        return MUTEX_CACHE[node_id]
    text = text_blob(node)
    labels: dict[str, str] = {}

    # Keep mutually exclusive measurements in separate synthetic patients.
    if "250908004" in snomed_ids(node):
        labels["lvef_measurement"] = node_id
    if "pre-test likelihood" in text or "clinical likelihood" in text:
        labels["pretest_band"] = node_id
    if "syntax score" in text:
        labels["syntax_band"] = node_id

    if "hfref" in text or "heart failure with reduced" in text:
        labels["hf_phenotype"] = "reduced"
    elif "hfmref" in text or "mildly reduced" in text:
        labels["hf_phenotype"] = "mildly_reduced"
    elif "hfpef" in text or "normal ejection fraction" in text:
        labels["hf_phenotype"] = "preserved"

    original_normalized = normalized_original(node)
    if re.search(r"\b(?:ccs|chronic coronary syndrome)\b", text):
        labels["ccs_mention"] = node_id
    if re.search(r"\b(?:heart failure|hfref|hfmref|hfpef|hf|chronic hf)\b", original_normalized):
        labels["heart_failure_mention"] = node_id
    if "cabg" in text or "coronary artery bypass" in text:
        labels["cabg_history_mention"] = node_id

    if re.search(r"single[- ]vessel", text):
        labels["coronary_vessel_count"] = "single"
    elif re.search(r"(?:double|two)[- ]vessel", text):
        labels["coronary_vessel_count"] = "double"
    elif re.search(r"(?:three|triple)[- ]vessel", text):
        labels["coronary_vessel_count"] = "triple"
    elif re.search(r"multivessel|multi vessel", text):
        labels["coronary_vessel_count"] = "multiple"

    if re.search(r"not involving the proximal lad", text):
        labels["proximal_lad_involvement"] = "absent"
    elif re.search(r"involving the proximal lad", text):
        labels["proximal_lad_involvement"] = "present"

    if re.search(r"ano?ca|inoca|non-obstructive|normal epicardial", text):
        labels["coronary_obstruction"] = "nonobstructive"
    elif re.search(
        r"obstructive cad|obstructive coronary|significant .*vessel|"
        r"functionally significant .*vessel|left main .*stenosis",
        text,
    ):
        labels["coronary_obstruction"] = "obstructive"

    if node_id in {"input_249210d3c9d81644ff1f", "input_004d40b9acf8c220a4ad"}:
        labels["weight_category"] = "overweight"
    elif re.search(r"\bsevere obesity\b|\bobese\b", text):
        labels["weight_category"] = "obese"
    elif re.search(r"\boverweight\b|\bbmi\b", text):
        labels["weight_category"] = "overweight"

    if node_id in {
        "input_797bf330c5a46789dd2b",
        "input_6c2ab456c39505ea45a6",
        "input_f8956f1a23d56a23c04c",
        "input_45194e4ad3cf31d09308",
        "input_33398bb7dd57c11caae7",
        "input_a0cbe969d8c5b321ce7f",
    }:
        labels["lvef_state"] = "reduced"
    elif node_id in {
        "input_48c408b0d324656ea19c",
        "input_df5467d7dafc528f182d",
        "input_3fdaac319ac4883df3e5",
    }:
        labels["lvef_state"] = "preserved_or_above_35"

    if node_id == "input_339a073dac8ee026c13d":
        labels["revascularization_history"] = "absent"
    elif node.get("role") == "Procedure" and re.search(
        r"pci|cabg|revascular|stent", text
    ):
        labels["revascularization_history"] = "present"

    if re.search(r"surgical risk", text):
        labels["surgical_risk_mention"] = node_id

    if "pre-test likelihood" in text or "clinical likelihood" in text:
        labels["diagnostic_stage"] = "pretest"
    elif re.search(
        r"functionally significant|significant .*vessel|left main .*stenosis|"
        r"multivessel (?:cad|coronary)|obstructive cad|obstructive coronary",
        text,
    ) and "suspected" not in text:
        labels["diagnostic_stage"] = "established_anatomy"

    if "no clinical heart failure" in text:
        labels["heart_failure_state"] = "absent"
    elif re.search(r"heart failure|\bhf\b", text):
        labels["heart_failure_state"] = "present"

    if "asymptomatic" in text:
        labels["symptom_state"] = "absent"
    elif re.search(r"symptomatic|angina|chest pain|dyspn|dizziness", text):
        labels["symptom_state"] = "present"

    if "sinus rhythm" in text:
        labels["rhythm_state"] = "sinus"
    elif "atrial fibrillation" in text or node["entity_original"].casefold() == "af":
        labels["rhythm_state"] = "atrial_fibrillation"

    if "not operable" in text or "inoperable" in text:
        labels["operability"] = "inoperable"
    elif "low surgical risk" in text or "surgically eligible" in text or "operable" in text:
        labels["operability"] = "operable"

    if "high surgical risk" in text or "very high surgical risk" in text:
        labels["surgical_risk"] = "high"
    elif "low surgical risk" in text:
        labels["surgical_risk"] = "low"

    if node_id == "input_791160d7a235b006ed01":
        labels["oac_indication"] = "absent"
    elif "indication for oac" in text or "indication for oral anticoagulation" in text:
        labels["oac_indication"] = "present"

    if "non-invasive imaging tests are unavailable" in text:
        labels["noninvasive_imaging"] = "unavailable"
    elif "functional imaging tests are available" in text or "ccta is available" in text:
        labels["noninvasive_imaging"] = "available"
    MUTEX_CACHE[node_id] = labels
    return labels


def nodes_compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    key = tuple(sorted((a["node_id"], b["node_id"])))
    if key in COMPATIBILITY_CACHE:
        return COMPATIBILITY_CACHE[key]
    result = True
    if normalized_original(a) and normalized_original(a) == normalized_original(b):
        result = False
    elif normalized_clause(a) and normalized_clause(a) == normalized_clause(b):
        result = False
    elif snomed_ids(a) and snomed_ids(a).intersection(snomed_ids(b)):
        result = False
    else:
        a_mutex = mutex_labels(a)
        b_mutex = mutex_labels(b)
        for family, value in a_mutex.items():
            if family in b_mutex and b_mutex[family] != value:
                result = False
                break
            if family in {
                "lvef_measurement",
                "pretest_band",
                "syntax_band",
                "ccs_mention",
                "heart_failure_mention",
                "cabg_history_mention",
                "surgical_risk_mention",
            } and family in b_mutex:
                result = False
                break
    COMPATIBILITY_CACHE[key] = result
    return result


def compatible(group: list[dict[str, Any]], node: dict[str, Any]) -> bool:
    return all(nodes_compatible(other, node) for other in group)


PAIR_SCORE_CACHE: dict[tuple[str, str], float] = {}


def pair_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    key = tuple(sorted((a["node_id"], b["node_id"])))
    if key in PAIR_SCORE_CACHE:
        return PAIR_SCORE_CACHE[key]
    rows_a = {row_id(occ) for occ in a.get("occurrences", [])}
    rows_b = {row_id(occ) for occ in b.get("occurrences", [])}
    files_a = {occ["file"] for occ in a.get("occurrences", [])}
    files_b = {occ["file"] for occ in b.get("occurrences", [])}
    score = 240.0 * len(rows_a.intersection(rows_b))
    score += 22.0 * len(files_a.intersection(files_b))
    score += 7.0 * len(category_tags(a).intersection(category_tags(b)))
    if a.get("role") == b.get("role"):
        score += 2.0
    PAIR_SCORE_CACHE[key] = score
    return score


def group_score(group: list[dict[str, Any]]) -> float:
    return sum(
        pair_score(group[i], group[j])
        for i in range(len(group))
        for j in range(i + 1, len(group))
    )


def build_groups(nodes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    four_node_case_count = len(nodes) - 70 * 3
    if not 0 <= four_node_case_count <= 70:
        raise ValueError(f"Cannot distribute {len(nodes)} nodes across 70 cases")
    target_sizes = [4] * four_node_case_count + [3] * (70 - four_node_case_count)
    assert sum(target_sizes) == len(nodes)
    rng = random.Random(20260801)

    # Seed groups with high-connectivity nodes, then fill each group with the
    # strongest compatible partner. Several deterministic restarts avoid a
    # poor early choice while keeping the output reproducible.
    relation_degree = {
        n["node_id"]: sum(pair_score(n, other) > 20 for other in nodes if other is not n)
        for n in nodes
    }
    best_groups: list[list[dict[str, Any]]] | None = None
    best_score = -math.inf

    for restart in range(6):
        jitter = {n["node_id"]: rng.random() for n in nodes}
        ordered = sorted(
            nodes,
            key=lambda n: (-relation_degree[n["node_id"]], jitter[n["node_id"]]),
        )
        if restart:
            # Preserve a bias toward connected seeds but explore alternatives.
            head = ordered[:140]
            rng.shuffle(head)
            ordered = head + ordered[140:]

        groups = [[] for _ in target_sizes]
        unassigned = {n["node_id"]: n for n in nodes}

        # One seed per group.
        for group in groups:
            seed = next(n for n in ordered if n["node_id"] in unassigned)
            group.append(seed)
            del unassigned[seed["node_id"]]

        failed = False
        while unassigned:
            open_group_indexes = [
                i for i, group in enumerate(groups) if len(group) < target_sizes[i]
            ]
            # Fill the most constrained group first.
            choices: list[tuple[int, int, float, str]] = []
            for group_index in open_group_indexes:
                group = groups[group_index]
                candidates = [n for n in unassigned.values() if compatible(group, n)]
                if not candidates:
                    failed = True
                    break
                best_candidate_score = max(
                    sum(pair_score(n, member) for member in group) for n in candidates
                )
                choices.append((len(candidates), group_index, -best_candidate_score, ""))
            if failed:
                break
            _, group_index, _, _ = min(choices)
            group = groups[group_index]
            candidates = [n for n in unassigned.values() if compatible(group, n)]
            candidates.sort(
                key=lambda n: (
                    -sum(pair_score(n, member) for member in group),
                    jitter[n["node_id"]],
                    n["node_id"],
                )
            )
            chosen = candidates[0]
            group.append(chosen)
            del unassigned[chosen["node_id"]]

        if failed:
            continue

        # Safe pairwise swaps improve topical cohesion without introducing a
        # mutually exclusive state.
        for _ in range(2_000):
            i, j = rng.sample(range(len(groups)), 2)
            ai = rng.randrange(len(groups[i]))
            bj = rng.randrange(len(groups[j]))
            before = group_score(groups[i]) + group_score(groups[j])
            candidate_i = groups[i].copy()
            candidate_j = groups[j].copy()
            candidate_i[ai], candidate_j[bj] = candidate_j[bj], candidate_i[ai]
            if not all(compatible(candidate_i[:k] + candidate_i[k + 1 :], n) for k, n in enumerate(candidate_i)):
                continue
            if not all(compatible(candidate_j[:k] + candidate_j[k + 1 :], n) for k, n in enumerate(candidate_j)):
                continue
            after = group_score(candidate_i) + group_score(candidate_j)
            if after > before:
                groups[i], groups[j] = candidate_i, candidate_j

        score = sum(group_score(group) for group in groups)
        if score > best_score:
            best_groups, best_score = groups, score

    if best_groups is None:
        raise RuntimeError("Unable to partition nodes into conflict-free groups")

    # Stable case numbering based on source location and node IDs.
    best_groups.sort(
        key=lambda group: min(
            (
                occ["file"],
                occ["table_id"],
                occ["row_index_zero_based"],
                node["node_id"],
            )
            for node in group
            for occ in node.get("occurrences", [])
        )
    )
    return best_groups


def apply_manual_group_swaps(
    groups: list[list[dict[str, Any]]], *, start_index: int = 0
) -> None:
    locations: dict[str, tuple[int, int]] = {}
    for group_index, group in enumerate(groups):
        for node_index, node in enumerate(group):
            locations[node["node_id"]] = (group_index, node_index)
    for left_id, right_id in MANUAL_GROUP_SWAPS[start_index:]:
        left_group, left_index = locations[left_id]
        right_group, right_index = locations[right_id]
        groups[left_group][left_index], groups[right_group][right_index] = (
            groups[right_group][right_index],
            groups[left_group][left_index],
        )
        locations[left_id] = (right_group, right_index)
        locations[right_id] = (left_group, left_index)

    for group in groups:
        for index, node in enumerate(group):
            if not compatible(group[:index] + group[index + 1 :], node):
                raise ValueError(
                    f"Manual grouping created an incompatible case around {node['node_id']}"
                )


def clean_context(text: str) -> str:
    text = text.strip().replace("%%", "%")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:In |in )?(?:selected )?(?:patients|individuals) (?:with|who|for whom) ", "", text)
    text = re.sub(r"^(?:patients|individuals) with ", "", text)
    text = re.sub(r"^the presence of specific comorbidities, such as ", "", text)
    return text.strip(" ,.;")


def clause_for_node(node: dict[str, Any]) -> str:
    node_id = node["node_id"]
    if node_id in CLAUSE_OVERRIDES:
        return CLAUSE_OVERRIDES[node_id]

    original = node["entity_original"].strip().replace("reurrent", "recurrent")
    original = re.sub(
        r"^(?:Patient with |patients with |patients in whom |individuals with )",
        "",
        original,
        flags=re.IGNORECASE,
    )
    original = re.sub(r"^in (?=drug-eluting)", "", original, flags=re.IGNORECASE)
    original = re.sub(r"\s+", " ", original).strip(" ,.;")
    return original


def join_clauses(clauses: Iterable[str]) -> str:
    cleaned = [clause.strip().rstrip(".") for clause in clauses]
    if any(any(separator in clause for separator in ":;()") for clause in cleaned):
        raise ValueError(f"Non-sentential separator in patient clause: {cleaned}")
    if len(cleaned) == 1:
        body = cleaned[0]
    elif len(cleaned) == 2:
        body = f"{cleaned[0]} and {cleaned[1]}"
    else:
        body = ", ".join(cleaned[:-1]) + f" and {cleaned[-1]}"
    return f"Patient with {body}."


def build_case(
    *,
    case_id: str,
    lexical_tier: str,
    patient_case: str,
    node_ids: list[str],
    node_by_id: dict[str, dict[str, Any]],
    lexical_provenance: dict[str, Any] | None = None,
    mention_overrides: dict[str, str] | None = None,
    derived_from_case_id: str | None = None,
) -> dict[str, Any]:
    selected = [node_by_id[node_id] for node_id in node_ids]
    row_to_nodes: dict[str, set[str]] = defaultdict(set)
    row_metadata: dict[str, dict[str, Any]] = {}
    intended_nodes: list[dict[str, Any]] = []

    for node in selected:
        source_rows = []
        for occurrence in node.get("occurrences", []):
            rid = row_id(occurrence)
            source_rows.append(rid)
            row_to_nodes[rid].add(node["node_id"])
            row_metadata[rid] = {
                "row_id": rid,
                "file": occurrence["file"],
                "table_id": occurrence["table_id"],
                "row_index_zero_based": occurrence["row_index_zero_based"],
            }
        intended_nodes.append(
            {
                "node_id": node["node_id"],
                "mention": (mention_overrides or {}).get(
                    node["node_id"], clause_for_node(node)
                ),
                "entity_original": node["entity_original"],
                "context": node.get("context", ""),
                "role": node.get("role"),
                "annotation_status": node.get("annotation_status"),
                "entity_standardized_list": node.get("entity_standardized_list", []),
                "logic_operators": sorted(
                    {item.get("operator") for item in node.get("logic_examples", [])}
                ),
                "source_row_ids": sorted(set(source_rows)),
            }
        )

    triggered_rows = []
    for rid in sorted(row_to_nodes):
        item = dict(row_metadata[rid])
        item["matched_node_ids"] = sorted(row_to_nodes[rid])
        triggered_rows.append(item)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "lexical_tier": lexical_tier,
        "patient_case": patient_case,
        "intended_node_ids": node_ids,
        "intended_nodes": intended_nodes,
        "trigger_semantics": TRIGGER_SEMANTICS,
        "expected_triggered_row_ids": [row["row_id"] for row in triggered_rows],
        "triggered_rows": triggered_rows,
        "expected_fully_satisfied_row_ids": None,
    }
    if lexical_provenance is not None:
        record["lexical_provenance"] = lexical_provenance
    if derived_from_case_id is not None:
        record["derived_from_case_id"] = derived_from_case_id
    return record


def synonym_provenance(item: dict[str, Any]) -> dict[str, Any]:
    concept_id = item["concept_id"]
    return {
        "mode": "snomed_ct_synonym",
        "terminology_source": "SNOMED CT US Edition",
        "terminology_view": "NCBO BioPortal (or NCBI MedGen where noted)",
        "concept_id": concept_id,
        "preferred_term": item["preferred_term"],
        "synonym_used": item["synonym_used"],
        "canonical_concept_uri": f"http://snomed.info/id/{concept_id}",
        "snomed_browser_url": (
            "https://browser.ihtsdotools.org/?perspective=full&conceptId1=" + concept_id
        ),
        "verification_url": item["verification_url"],
        "verified_on": "2026-08-01",
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = json.loads(args.input_nodes.read_text(encoding="utf-8"))
    all_nodes = source["unique_combinations"]
    node_by_id = {node["node_id"]: node for node in all_nodes}
    if len(node_by_id) != 277:
        raise ValueError(f"Expected 277 unique input nodes, found {len(node_by_id)}")
    unknown_exclusions = sorted(set(EXCLUDED_NODES).difference(node_by_id))
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded node IDs: {unknown_exclusions}")
    nodes = [node for node in all_nodes if node["node_id"] not in EXCLUDED_NODES]
    if len(nodes) != 249:
        raise ValueError(f"Expected 249 clinically usable nodes, found {len(nodes)}")

    groups = build_groups(nodes)
    apply_manual_group_swaps(groups)
    records: list[dict[str, Any]] = []
    role_order = {"ClinicalCondition": 0, "ClinicalParameter": 1, "Medication": 2, "Procedure": 3, "string": 4}
    for number, group in enumerate(groups, start=1):
        group = sorted(
            group,
            key=lambda node: (
                role_order.get(node.get("role", ""), 9),
                clause_for_node(node).casefold(),
                node["node_id"],
            ),
        )
        records.append(
            build_case(
                case_id=f"near_{number:03d}",
                lexical_tier="near_original_or_snomed_preferred",
                patient_case=join_clauses(clause_for_node(node) for node in group),
                node_ids=[node["node_id"] for node in group],
                node_by_id=node_by_id,
            )
        )

    base_records = list(records)
    base_by_node = {
        node_id: record
        for record in base_records
        for node_id in record["intended_node_ids"]
    }

    for number, item in enumerate(SNOMED_VARIANTS, start=1):
        target_id = item["target_node_id"]
        source_record = base_by_node[target_id]
        source_mentions = {
            node["node_id"]: node["mention"] for node in source_record["intended_nodes"]
        }
        replacement = item["replacement"]
        variant_mentions = dict(source_mentions)
        variant_mentions[target_id] = replacement
        provenance = synonym_provenance(item)
        provenance.update(
            {
                "target_node_id": target_id,
                "base_mention": source_mentions[target_id],
                "replacement_mention": replacement,
            }
        )
        records.append(
            build_case(
                case_id=f"snomed_syn_{number:03d}",
                lexical_tier="snomed_ct_synonym",
                patient_case=join_clauses(
                    variant_mentions[node_id]
                    for node_id in source_record["intended_node_ids"]
                ),
                node_ids=list(source_record["intended_node_ids"]),
                node_by_id=node_by_id,
                lexical_provenance=provenance,
                mention_overrides={target_id: replacement},
                derived_from_case_id=source_record["case_id"],
            )
        )

    for number, item in enumerate(FREE_VARIANTS, start=1):
        target_id = item["target_node_id"]
        source_record = base_by_node[target_id]
        source_mentions = {
            node["node_id"]: node["mention"] for node in source_record["intended_nodes"]
        }
        replacement = item["replacement"]
        variant_mentions = dict(source_mentions)
        variant_mentions[target_id] = replacement
        records.append(
            build_case(
                case_id=f"free_syn_{number:03d}",
                lexical_tier="free_paraphrase",
                patient_case=join_clauses(
                    variant_mentions[node_id]
                    for node_id in source_record["intended_node_ids"]
                ),
                node_ids=list(source_record["intended_node_ids"]),
                node_by_id=node_by_id,
                lexical_provenance={
                    "mode": "free_paraphrase",
                    "note": "Deliberately freer wording; not asserted as a SNOMED CT synonym.",
                    "target_node_id": target_id,
                    "base_mention": source_mentions[target_id],
                    "replacement_mention": replacement,
                },
                mention_overrides={target_id: replacement},
                derived_from_case_id=source_record["case_id"],
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "gold_standard_trigger_cases.json", records)
    with (args.output_dir / "gold_standard_trigger_cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    compact_records = [
        {
            "case_id": record["case_id"],
            "lexical_tier": record["lexical_tier"],
            "patient_case": record["patient_case"],
            "intended_node_ids": record["intended_node_ids"],
            "expected_triggered_row_ids": record["expected_triggered_row_ids"],
            "expected_fully_satisfied_row_ids": None,
            **(
                {"derived_from_case_id": record["derived_from_case_id"]}
                if "derived_from_case_id" in record
                else {}
            ),
        }
        for record in records
    ]
    with (args.output_dir / "gold_standard_trigger_cases.compact.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in compact_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    base_counts = Counter(
        node_id
        for record in records[:70]
        for node_id in record["intended_node_ids"]
    )
    all_input_ids = set(node_by_id)
    included_input_ids = all_input_ids.difference(EXCLUDED_NODES)
    duplicate_base_ids = sorted(node_id for node_id, count in base_counts.items() if count != 1)
    missing_base_ids = sorted(included_input_ids.difference(base_counts))
    tier_counts = Counter(record["lexical_tier"] for record in records)
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "case_count": len(records),
        "tier_counts": dict(sorted(tier_counts.items())),
        "input_node_count": len(all_input_ids),
        "clinically_usable_input_node_count": len(included_input_ids),
        "excluded_input_node_count": len(EXCLUDED_NODES),
        "base_70_unique_node_count": len(base_counts),
        "base_70_covers_every_clinically_usable_node_exactly_once": (
            set(base_counts) == included_input_ids
            and not duplicate_base_ids
            and not missing_base_ids
        ),
        "base_70_duplicate_or_multiply_used_node_ids": duplicate_base_ids,
        "base_70_missing_included_node_ids": missing_base_ids,
        "excluded_input_node_ids": sorted(EXCLUDED_NODES),
        "distinct_triggered_row_count_all_cases": len(
            {
                row_id
                for record in records
                for row_id in record["expected_triggered_row_ids"]
            }
        ),
        "distinct_triggered_row_count_base_70": len(
            {
                row_id
                for record in records[:70]
                for row_id in record["expected_triggered_row_ids"]
            }
        ),
        "fully_satisfied_rows_adjudicated": False,
    }
    write_json(args.output_dir / "coverage_report.json", coverage)
    write_json(
        args.output_dir / "excluded_input_nodes.json",
        [
            {
                "node_id": node_id,
                "reason": EXCLUDED_NODES[node_id],
                "entity_original": node_by_id[node_id]["entity_original"],
                "context": node_by_id[node_id].get("context", ""),
                "role": node_by_id[node_id].get("role"),
                "source_row_ids": sorted(
                    row_id(occurrence)
                    for occurrence in node_by_id[node_id].get("occurrences", [])
                ),
            }
            for node_id in sorted(EXCLUDED_NODES)
        ],
    )
    write_json(
        args.output_dir / "snomed_synonym_provenance.json",
        [record for record in records if record["lexical_tier"] == "snomed_ct_synonym"],
    )
    write_json(
        args.output_dir / "generation_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_on": "2026-08-01",
            "source_input_nodes_sha256": hashlib.sha256(
                args.input_nodes.read_bytes()
            ).hexdigest(),
            "source_schema_version": source.get("schema_version"),
            "source_unique_node_count": source.get("total_unique_combinations"),
            "source_occurrence_count": source.get("total_node_occurrences"),
            "included_input_node_count": len(included_input_ids),
            "excluded_input_node_count": len(EXCLUDED_NODES),
            "case_count": len(records),
            "trigger_semantics": TRIGGER_SEMANTICS,
        },
    )

    with (args.output_dir / "case_index.tsv").open("w", encoding="utf-8") as handle:
        handle.write("case_id\tlexical_tier\tpatient_case\tintended_node_ids\texpected_triggered_row_ids\n")
        for record in records:
            handle.write(
                "\t".join(
                    [
                        record["case_id"],
                        record["lexical_tier"],
                        record["patient_case"].replace("\t", " "),
                        "|".join(record["intended_node_ids"]),
                        "|".join(record["expected_triggered_row_ids"]),
                    ]
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
