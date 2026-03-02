# Migrate to GLM 4.6

GLM 4.6 is a foundation model from Zhipu AI (Z.ai) designed for coding and agentic workflows. Its key strengths include tool use, coding tasks, reasoning, and knowledge retrieval. Developers commonly migrate to GLM 4.6 for its lower cost, high output speeds, and strong coding and reasoning performance. However, this model behaves differently from proprietary models such as Claude Sonnet or GPT-4-series models. Reusing existing prompts without adjustment may lead to degraded output quality. This guide includes specific recommendations for adapting prompts and workflows.

## Model Overview

*   **Architecture**: Built on the “GLM-4.x” foundation, leveraging a Mixture-of-Experts (MoE) Transformer architecture.
*   **Efficiency**: The model has 355 billion parameters in total. However, thanks to MoE sparsity, only about 32 billion parameters are active on any given forward pass, yielding considerable efficiency gains.
*   **Open Source**: GLM 4.6 is released under an MIT License, giving you the flexibility to fine-tune, self-host, or deploy however you choose.
*   **Data Privacy**: When you run GLM 4.6 on providers like Cerebras Inference, your data is never used to train new models or retained after processing.

For complete model specifications, pricing details, and rate limits, see the GLM 4.6 model page.

## Migration Best Practices

When migrating to GLM 4.6, a common mistake is reusing old prompts without adjusting them for its unique behavior, which can lead to suboptimal performance. To fully leverage this model’s strengths, it’s essential to refine your prompts, architecture, and sampling parameters accordingly.

### 1. Front-load instructions

GLM 4.6 places heightened attention on the beginning of the prompt. To ensure consistent instruction following, place all required rules, constraints, and behavioral instructions at the beginning of the system prompt. This is especially important when using prompting patterns that rely on “think” tags.

### 2. Use clear and direct instructions

GLM 4.6 responds more reliably to explicit rules than to suggestive or optional language.

*   Use unambiguous terms such as MUST, REQUIRED, or STRICTLY.
*   Avoid soft phrasing such as “Please try to…” or indirect suggestions.

For example:

*   **Do**: “Before writing any code, you MUST first read and fully comprehend the architecture.md file. All code you generate must strictly conform…”
*   **Don’t**: “Please read and follow my architecture.md…“

### 3. Specify a default language

Because GLM 4.6 is multilingual, it may occasionally switch languages if not instructed otherwise. We’ve also observed that the model may output reasoning traces in Chinese on the first turn. Explicit language control prevents this behavior. Add a directive like “Always respond in English” (or your preferred language) in your system prompt to prevent unexpected responses or reasoning traces in other languages.

### 4. Use role prompts intentionally

GLM 4.6 follows roles and personas closely. Assigning clear roles improves consistency and accuracy. Example: "You are a senior software architect. Review the following specifications and produce a structured design proposal." Role-based prompting also works well in multi-agent systems, with each agent having its own persona.

### 5. Use critic agents for validation

When building agentic systems, rather than relying on a single agent to both generate and validate code, create dedicated critics to review and validate outputs before allowing the main agentic flow to advance in its plan. These could include:

*   **Code reviewer**: A sub-agent configured to rigorously check for code quality, adherence to SOLID/DRY/YAGNI principles, and maintainability issues.
*   **QA tester**: Potentially bound with agentic browser capabilities to test user flows, edge cases, and integration points.
*   **Security reviewer**: Specialized in identifying vulnerabilities, unsafe patterns, and compliance issues.
*   **Performance analyst**: Focused on detecting performance bottlenecks, inefficient algorithms, or resource leaks.

This pattern improves reliability and aligns well with GLM 4.6’s behavior. Multi-agent frameworks like Code Puppy, Kilo/Roo Code, and others support this approach.

### 6. Break down tasks

GLM 4.6 performs a single reasoning pass per prompt and does not continuously re-evaluate mid-task. This is referred to as interleaved thinking, which is supported by other Sonnet and OpenAI models. Without interleaved thinking, we recommend breaking tasks into small, well-defined substeps to prompt better task completion. For example:

1.  List dependencies
2.  Propose new structure
3.  Generate code
4.  Verify output

### 7. Minimize reasoning when not needed

GLM 4.6 may generate verbose reasoning blocks that are unnecessary and slow down responses. We recommend the following:

*   **Disable Reasoning** with the nonstandard `disable_reasoning: True` parameter. See our reasoning guide for more information. This is different from the thinking parameter that Z.ai uses in their API.
*   Set appropriate `max_completion_tokens` limits. For focused responses, consider using lower values.
*   Use prompt-based control by adding instructions to minimize reasoning in your system prompt. For example: “Reason only when necessary” or “Skip reasoning for straightforward tasks.”
*   Use structured output formats (JSON, lists, bullets) that naturally discourage verbose reasoning blocks.

### 8. Enable enhanced reasoning for complex tasks

For tasks requiring deeper analysis:

*   Ensure `disable_reasoning` is false or omitted.
*   Add reasoning directives such as:
    *   “Think step by step.”
    *   “Break the problem down logically.”
*   Include examples that demonstrate the reasoning process you want, showing the model how to work through problems methodically.

### 9. Combine GLM 4.6 with frontier models when needed

If your workload includes tasks requiring frontier-level reasoning accuracy, consider hybrid architectures:

1.  Route simpler tasks to GLM 4.6 and use a frontier model for more complex queries.
2.  Use GLM 4.6 as a fast agent that loops in frontier models only when needed.
3.  Use a frontier model to create a plan, then execute it rapidly with GLM 4.6.

This approach reduces cost and latency while maintaining high accuracy where required.

### 10. Tune sampling parameters

Parameter tuning has a significant impact on output quality. The recommended defaults from Z.ai and Cerebras are:

| Parameter | Recommended Range | Notes |
| :--- | :--- | :--- |
| `temperature` | 1.0 (general) / 0.6 (instruction following) | Very low values may degrade output quality. |
| `top_p` | 0.95 | Balanced default. |

On Cerebras, adjust these parameters via the API:

```python
completion_create_response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Explain how photosynthesis works."}],
    model="zai-glm-4.6",
    stream=False,
    max_completion_tokens=65536,
    temperature=1,
    top_p=1,
)
```

## Next Steps

*   View GLM 4.6 model info - Pricing, rate limits, and capabilities
*   Get an API key - Test GLM 4.6 in our API playground
*   Join the Cerebras Discord - Share feedback, observations, and best practices with other developers
