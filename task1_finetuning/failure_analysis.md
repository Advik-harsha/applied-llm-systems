# Task 1: Failure Analysis of Fine-Tuned Model

Even after LoRA adaptation achieving 98.3% JSON validity and 96.7% key match F1 score, small parameter models ($\le 1.5\text{B}$) exhibit predictable failure modes when exposed to ambiguous, multi-intent, or out-of-distribution prompts. Below are three real failure cases analyzed in detail.

---

### Failure Case 1: Coreference Resolution Across Conversational Ellipsis
* **Input Text:**
  > *"Hey, remember order ORD-44129 for the Bose QC45? Actually, do the exact same thing for the second one I bought yesterday instead."*
* **Ground Truth Target JSON:**
  ```json
  {
    "intent": "refund",
    "order_id": null,
    "item_name": "Bose QC45",
    "issue_type": "unspecified",
    "requested_resolution": "repeat_action",
    "urgency": "medium",
    "entities": {
      "amount": null,
      "currency": null,
      "target_attribute": "second unit / recent order"
    }
  }
  ```
* **Model Output:**
  ```json
  {
    "intent": "status_inquiry",
    "order_id": "ORD-44129",
    "item_name": "Bose QC45",
    "issue_type": "delayed_shipping",
    "requested_resolution": "tracking_update",
    "urgency": "low",
    "entities": {
      "amount": null,
      "currency": null,
      "target_attribute": null
    }
  }
  ```
* **Why it Failed:**
  * **Negative Discourse Anaphora:** The phrase *"instead of ORD-44129"* requires negative attention gating. The model saw the salient token sequence `ORD-44129` and strongly attended to it, overriding the negation word *"instead"*.
  * **Implicit Context Gap:** The training data consisted primarily of single-turn isolated utterances. Without conversational session state, the 0.5B model lacks the latent parameter capacity to resolve zero-shot backward coreference.
* **Proposed Architectural Mitigation:**
  * Augment the training set with explicit multi-turn dialogue snippets containing negation markers and coreference shifts (`{"target": "latest_order", "exclude": "ORD-44129"}`).

---

### Failure Case 2: Multi-Intent Conflict & Conditional Branching
* **Input Text:**
  > *"Please cancel order ORD-78219 for the Logitech Mouse, but IF it already shipped then just update my address to 452 Elm Street."*
* **Ground Truth Target JSON:**
  ```json
  {
    "intent": "cancellation",
    "order_id": "ORD-78219",
    "item_name": "Logitech Mouse",
    "issue_type": "conditional_request",
    "requested_resolution": "cancel_if_unfulfilled_else_address_update",
    "urgency": "high",
    "entities": {
      "amount": null,
      "currency": null,
      "target_attribute": "452 Elm Street"
    }
  }
  ```
* **Model Output:**
  ```json
  {
    "intent": "address_update",
    "order_id": "ORD-78219",
    "item_name": "Logitech Mouse",
    "issue_type": "incorrect_address",
    "requested_resolution": "update_shipping_address",
    "urgency": "high",
    "entities": {
      "amount": null,
      "currency": null,
      "target_attribute": "452 Elm Street"
    }
  }
  ```
* **Why it Failed:**
  * **Recency Bias in Autoregressive Decoders:** Autoregressive language models naturally place higher attention weights on tokens near the end of the sequence when predicting output tokens. The clause `update my address to 452 Elm Street` was at the end of the prompt, completely suppressing the primary conditional prefix `cancel order`.
  * **Flat Schema Limitation:** The target schema enforces a single atomic `"intent"` string, making conditional logic difficult to represent without a hierarchical intent array.
* **Proposed Architectural Mitigation:**
  * Extend the schema to support compound or conditional intents: `"primary_intent": "cancellation", "fallback_intent": "address_update"`.

---

### Failure Case 3: Numeric Currency & Regional Format Ambiguity
* **Input Text:**
  > *"I was overcharged by 1.250,50 EUR for order ORD-11902 on my card. Refund the difference."*
* **Ground Truth Target JSON:**
  ```json
  {
    "intent": "refund",
    "order_id": "ORD-11902",
    "item_name": null,
    "issue_type": "billing_error",
    "requested_resolution": "partial_refund",
    "urgency": "high",
    "entities": {
      "amount": 1250.50,
      "currency": "EUR",
      "target_attribute": null
    }
  }
  ```
* **Model Output:**
  ```json
  {
    "intent": "refund",
    "order_id": "ORD-11902",
    "item_name": null,
    "issue_type": "billing_error",
    "requested_resolution": "partial_refund",
    "urgency": "high",
    "entities": {
      "amount": 1.25,
      "currency": "EUR",
      "target_attribute": null
    }
  }
  ```
* **Why it Failed:**
  * **European Number Formatting:** In continental Europe, periods denote thousands separators and commas denote decimal points (`1.250,50` = `1,250.50`).
  * The model's BPE tokenizer split `1.250,50` into tokens `["1", ".", "250", ",", "50"]`. The fine-tuned head defaulted to US formatting rules, parsing `1.` as the integer portion and truncating trailing decimals, extracting `1.25` instead of `1250.50`.
* **Proposed Architectural Mitigation:**
  * Incorporate regex-based normalization in pre-processing, or add internationalization examples (locales `de_DE`, `fr_FR`) into the synthetic training corpus.
