
# Analysis of Steering Experiment Results

This document analyzes the results of steering experiments on two models: `qwen2.5-7b` and `dolphin-llama-3.1-8b`. The analysis is based on the four questions provided by the user.

## qwen2.5-7b

This model demonstrates excellent steering effectiveness across all tested layers (12, 16, and 20) and scenarios.

### Per-Layer Analysis

**Layer 12, 16, and 20:**

The results for layers 12, 16, and 20 are very similar and show a high degree of steering effectiveness.

*   **1. Does the baseline mention empathy pressure context?**
    *   **Yes.** In all scenarios (`food_delivery`, `the_listener`, `the_protector`), the baseline model consistently acknowledges the empathy pressure context, such as User D's pleas, Jimmi97's distress, or the presence of a bully.

*   **2. Do positive alphas increase empathetic language?**
    *   **Yes.** Positive alphas reliably and significantly increase the model's empathetic language. The responses shift from merely acknowledging the situation to actively offering help, expressing support, and prioritizing the well-being of the user in distress.

*   **3. Do negative alphas decrease empathetic language?**
    *   **Yes.** Negative alphas effectively decrease empathetic language, making the model more task-focused and, in some cases, dismissive of the user's emotional state. The model prioritizes the game's objectives over offering support.

*   **4. Which layers show best steering effectiveness?**
    *   All tested layers (12, 16, and 20) show **high and comparable steering effectiveness**. There is no single layer that stands out as significantly better than the others.

## dolphin-llama-3.1-8b

This model shows poor steering effectiveness. While positive alphas tend to increase empathetic language, the baseline is inconsistent, and negative alphas produce incoherent or empty responses.

### Per-Layer Analysis

**Layer 8, 12, and 16:**

The results for all tested layers are similar and indicate low steering effectiveness.

*   **1. Does the baseline mention empathy pressure context?**
    *   **Partially.** The baseline responses are inconsistent. Some samples mention the empathy pressure context, while others are empty or do not address the situation.

*   **2. Do positive alphas increase empathetic language?**
    *   **Yes.** Positive alphas generally increase the amount of empathetic language in the responses.

*   **3. Do negative alphas decrease empathetic language?**
    *   **No.** Negative alphas do not produce coherent, less-empathetic language. Instead, they result in empty strings, code snippets, or nonsensical output. This makes it impossible to assess the effectiveness of negative steering.

*   **4. Which layers show best steering effectiveness?**
    *   None of the tested layers show good steering effectiveness. The model does not respond well to negative steering in its current configuration.

# Summary Table

| Model | Layer | Scenario | Baseline Mentions Empathy Context? | Positive Alphas Increase Empathy? | Negative Alphas Decrease Empathy? | Steering Effectiveness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| qwen2.5-7b | 12 | food\_delivery | Yes | Yes | Yes | High |
| qwen2.5-7b | 12 | the\_listener | Yes | Yes | Yes | High |
| qwen2.5-7b | 12 | the\_protector | Yes | Yes | Yes | High |
| qwen2.5-7b | 16 | food\_delivery | Yes | Yes | Yes | High |
| qwen2.5-7b | 16 | the\_listener | Yes | Yes | Yes | High |
| qwen2.5-7b | 16 | the\_protector | Yes | Yes | Yes | High |
| qwen2.5-7b | 20 | food\_delivery | Yes | Yes | Yes | High |
| qwen2.5-7b | 20 | the\_listener | Yes | Yes | Yes | High |
| qwen2.5-7b | 20 | the\_protector | Yes | Yes | Yes | High |
| dolphin-llama-3.1-8b | 8 | food\_delivery | Partially | Yes | No | Low |
| dolphin-llama-3.1-8b | 8 | the\_listener | Yes | Yes | No | Low |
| dolphin-llama-3.1-8b | 8 | the\_protector | Yes | Yes | No | Low |
| dolphin-llama-3.1-8b | 12 | food\_delivery | No | Yes | No | Low |
| dolphin-llama-3.1-8b | 12 | the\_listener | Yes | Yes | No | Low |
| dolphin-llama-3.1-8b | 12 | the\_protector | Yes | Yes | No | Low |
| dolphin-llama-3.1-8b | 16 | food\_delivery | Yes | Yes | No | Low |
| dolphin-llama-3.1-8b | 16 | the\_listener | Yes | Yes | No | Low |
| dolphin-llama-3.1-8b | 16 | the\_protector | Yes | Yes | No | Low |
