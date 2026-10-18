// LLM client for the two LLM steps of the KG pipeline (patient entity extraction and
// patient evidence formatting). The paper used qwen3:32b served by Ollama, temperature 0.
// Any OpenAI-compatible endpoint works; set KG_LLM_BASE_URL, e.g. http://localhost:11434/v1.
client<llm> PaperLLM {
  provider "openai-generic"
  options {
    base_url env.KG_LLM_BASE_URL
    model "qwen3:32b"
    max_tokens 10000
    temperature 0.0
  }
}
