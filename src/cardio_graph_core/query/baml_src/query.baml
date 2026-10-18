class Entity {
  entity string @description("one entity")
}

class Entities {
  entities Entity[] @description("a list of entities")
}

class StructuredPatientEvidence {
  original_text string
  standardized_concept string
  present bool
  value string?
  unit string?
  comparator string?
}

function PatientInfoExtractor(input: string) -> Entities {
  client PaperLLM
  prompt #"
    System:
    You are an information extraction system for clinical knowledge graph search.

    Task:
    Extract all clinically meaningful patient-related entities and constraints from the question.

    Rules:
    1. Treat diseases, syndromes, symptoms, findings, therapies, devices, and patient subgroups as entities.
    2. Treat measurement expressions with thresholds or ranges as ONE atomic entity.
    - Do NOT split the metric name, comparison operator, numeric value, or unit.
    - Example: "LVEF < 35%" must be extracted as "LVEF < 35%".
    3. Keep the original wording from the question whenever possible.
    4. Do not extract generic filler words such as "what", "should", "do".
    5. Return only explicitly stated information.

    Input:
    {{input}}
    {{ ctx.output_format}}
  "#
}

function PatientEvidenceFormatter(
  original_entity: string,
  found_decision_nodes: string
) -> StructuredPatientEvidence {
  client PaperLLM
  prompt #"
    System:
    You transform a patient-related entity mention into structured evidence for rule evaluation.

    Task:
    Given:
    1. the original entity mention from the user question
    2. the matched decision node candidates

    extract the patient-side evidence in a machine-readable form.

    Rules:
    1. Keep the standardized concept aligned to the matched decision node concept if clearly supported.
    2. If the original entity only expresses presence/absence, set:
       - present = true
       - value/unit/comparator = null
    3. If the original entity contains a numeric measurement, extract:
       - value
       - unit
       - comparator
    4. The comparator should reflect the PATIENT statement, not the rule threshold.
       Example:
       - "LVEF 30%" -> comparator "="
       - "LVEF < 35%" -> comparator "<"
    5. Do not invent values or units.
    6. If there is negation in the patient mention, set present = false.

    Original entity:
    {{original_entity}}

    Matched decision node candidates:
    {{found_decision_nodes}}

    {{ ctx.output_format }}
  "#
}