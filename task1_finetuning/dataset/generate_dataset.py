"""
Dataset Generator for Natural Language Customer Support Intent to Strict JSON
Generates train (250 examples) and held-out test (60 examples) datasets.
Target Schema:
{
  "intent": "refund" | "exchange" | "address_update" | "cancellation" | "status_inquiry" | "escalation",
  "order_id": str | null,
  "item_name": str | null,
  "issue_type": str | null,
  "requested_resolution": str,
  "urgency": "low" | "medium" | "high",
  "entities": {
    "amount": float | null,
    "currency": str | null,
    "target_attribute": str | null
  }
}
"""

import json
import random
import os

TEMPLATES = [
    # Refund Intent
    {
        "intents": ["refund"],
        "patterns": [
            ("I bought {item} (Order {order_id}) last week but it stopped working after 2 days. Please issue a refund of {amount} {currency} back to my account.", "defective_product", "full_refund", "high"),
            ("Please return my money for order {order_id}. The {item} arrived damaged and shattered.", "damaged_item", "full_refund", "medium"),
            ("Hey, can I get a refund of {amount} {currency} on order #{order_id}? The {item} didn't fit at all.", "wrong_size", "full_refund", "low"),
            ("Received the wrong item for order {order_id}. Ordered {item} but received something totally different. I want my {amount} {currency} back immediately.", "wrong_item_received", "full_refund", "high"),
            ("I was double charged for order {order_id} ({item}). The extra charge was {amount} {currency}. Refund please.", "billing_error", "partial_refund", "high"),
        ]
    },
    # Exchange Intent
    {
        "intents": ["exchange"],
        "patterns": [
            ("Hello, I need to exchange the {item} from order {order_id} for a larger size {target_attribute}. How do I ship it back?", "size_mismatch", "size_exchange", "medium"),
            ("I got order {order_id} today, but the {item} is in black instead of {target_attribute}. Can we do a color swap?", "color_mismatch", "color_exchange", "low"),
            ("The {item} in order #{order_id} has a scratched lens. I don't want a refund, just send a replacement unit.", "damaged_item", "replacement_unit", "medium"),
            ("Need to replace my {item} (Order: {order_id}) with model {target_attribute} if possible.", "model_upgrade", "product_exchange", "medium"),
        ]
    },
    # Cancellation Intent
    {
        "intents": ["cancellation"],
        "patterns": [
            ("Cancel order {order_id} immediately! I accidentally ordered the wrong {item}.", "accidental_order", "order_cancellation", "high"),
            ("I would like to cancel my order #{order_id} for {item}. Found it cheaper elsewhere.", "found_cheaper_alternative", "order_cancellation", "low"),
            ("Please abort shipment on order {order_id} ({item}). I no longer need it.", "changed_mind", "order_cancellation", "medium"),
        ]
    },
    # Status Inquiry Intent
    {
        "intents": ["status_inquiry"],
        "patterns": [
            ("Where is my package? Order number {order_id} containing {item} was supposed to arrive yesterday.", "delayed_shipping", "tracking_update", "high"),
            ("Can you check the dispatch status for order {order_id} ({item})?", "shipping_inquiry", "status_check", "low"),
            ("Has order #{order_id} ({item}) been shipped yet? Tracking has had no movement for 3 days.", "stuck_in_transit", "tracking_investigation", "medium"),
        ]
    },
    # Address Update Intent
    {
        "intents": ["address_update"],
        "patterns": [
            ("I moved! Please change the delivery address for order {order_id} ({item}) to {target_attribute}.", "incorrect_address", "update_shipping_address", "high"),
            ("Typo in shipping address for order #{order_id}. Apartment should be {target_attribute}. Please update before dispatch.", "address_typo", "update_shipping_address", "high"),
        ]
    },
    # Escalation Intent
    {
        "intents": ["escalation"],
        "patterns": [
            ("Your customer support has ignored me for a week regarding order {order_id} ({item}). Escalate this to a manager right now!", "poor_service", "managerial_escalation", "high"),
            ("I am taking legal action if order {order_id} for {amount} {currency} is not resolved today. Connect me to your supervisor.", "unresolved_dispute", "legal_escalation", "high"),
        ]
    }
]

ITEMS = [
    "Sony WH-1000XM5 Headphones", "Logitech MX Master 3S Mouse", "MacBook Pro 14 M3",
    "Nike Air Zoom Pegasus 40", "Samsung Galaxy S24 Ultra", "Dell UltraSharp 27 4K Monitor",
    "Kindle Paperwhite 16GB", "Keychron Q1 Pro Mechanical Keyboard", "Anker 737 Power Bank",
    "Dyson V12 Cordless Vacuum", "Bose QuietComfort 45", "Apple Watch Series 9"
]

CURRENCIES = ["USD", "EUR", "GBP", "INR", "CAD"]
ATTRIBUTES = ["Size XL", "Navy Blue", "Space Gray", "128 Baker St, Apt 4B", "Size 10.5 US", "Forest Green"]

def generate_record(idx, seed_val):
    random.seed(seed_val)
    group = random.choice(TEMPLATES)
    intent = group["intents"][0]
    pattern, issue_type, resolution, urgency = random.choice(group["patterns"])
    
    order_id = f"ORD-{random.randint(10000, 99999)}"
    item = random.choice(ITEMS)
    amount = round(random.uniform(25.0, 850.0), 2)
    currency = random.choice(CURRENCIES)
    target_attr = random.choice(ATTRIBUTES)
    
    # Natural variations / noise
    text = pattern.format(
        item=item,
        order_id=order_id,
        amount=amount,
        currency=currency,
        target_attribute=target_attr
    )
    
    # Random typo or case alteration
    if random.random() < 0.2:
        text = text.lower()
    
    has_amount = "{amount}" in pattern
    has_target = "{target_attribute}" in pattern
    
    target_json = {
        "intent": intent,
        "order_id": order_id,
        "item_name": item,
        "issue_type": issue_type,
        "requested_resolution": resolution,
        "urgency": urgency,
        "entities": {
            "amount": amount if has_amount else None,
            "currency": currency if has_amount else None,
            "target_attribute": target_attr if has_target else None
        }
    }
    
    system_prompt = (
        "You are an AI assistant for e-commerce customer support intent parsing. "
        "Extract the customer's request into strict JSON conforming to the schema. "
        "Do not include any conversational filler or explanation."
    )
    
    return {
        "id": f"sample_{idx:04d}",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(target_json, indent=2)}
        ],
        "input_text": text,
        "target_json": target_json
    }

def main():
    os.makedirs("task1_finetuning/dataset", exist_ok=True)
    
    train_records = [generate_record(i, seed_val=1000 + i) for i in range(250)]
    test_records = [generate_record(i, seed_val=5000 + i) for i in range(60)]
    
    with open("task1_finetuning/dataset/train.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
            
    with open("task1_finetuning/dataset/test.jsonl", "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Generated {len(train_records)} training records and {len(test_records)} test records.")

if __name__ == "__main__":
    main()
