#!/usr/bin/env python3
"""
Generate sample datasets for testing and demo purposes.

Produces:
  data/orders.csv      — 100K rows, e-commerce orders
  data/customers.csv   — 10K rows, customer records
  data/products.csv    — 1K rows, product catalogue
  data/events.jsonl    — 50K rows, user event log
"""
import csv
import json
import os
import random
import time

random.seed(42)

COUNTRIES = ["CA", "US", "UK", "DE", "FR", "AU", "JP", "BR", "MX", "IN"]
CATEGORIES = ["electronics", "clothing", "home", "books", "sports", "food", "beauty", "toys"]
STATUSES = ["completed", "pending", "cancelled", "refunded"]
EVENT_TYPES = ["page_view", "add_to_cart", "checkout", "purchase", "search", "login"]

os.makedirs("data", exist_ok=True)


def gen_orders(n: int = 100_000):
    path = "data/orders.csv"
    print(f"  Generating {n:,} orders → {path}")
    t = time.time()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "customer_id", "product_id", "country",
                    "category", "quantity", "unit_price", "total", "status", "created_at"])
        for i in range(1, n + 1):
            qty = random.randint(1, 10)
            price = round(random.uniform(5.0, 500.0), 2)
            w.writerow([
                i,
                random.randint(1, 10_000),
                random.randint(1, 1_000),
                random.choice(COUNTRIES),
                random.choice(CATEGORIES),
                qty,
                price,
                round(qty * price, 2),
                random.choice(STATUSES),
                f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            ])
    print(f"    done in {time.time()-t:.1f}s ({os.path.getsize(path)/1024/1024:.1f} MB)")


def gen_customers(n: int = 10_000):
    path = "data/customers.csv"
    print(f"  Generating {n:,} customers → {path}")
    t = time.time()
    first = ["Alice","Bob","Carlos","Diana","Eva","Frank","Grace","Hiro","Iris","James"]
    last  = ["Smith","Jones","Garcia","Kim","Chen","Müller","Singh","Okafor","Martin","Nakamura"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id","name","country","email","age","is_premium"])
        for i in range(1, n + 1):
            w.writerow([
                i,
                f"{random.choice(first)} {random.choice(last)}",
                random.choice(COUNTRIES),
                f"user{i}@example.com",
                random.randint(18, 75),
                random.choice(["true","false"]),
            ])
    print(f"    done in {time.time()-t:.1f}s")


def gen_products(n: int = 1_000):
    path = "data/products.csv"
    print(f"  Generating {n:,} products → {path}")
    t = time.time()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product_id","name","category","price","stock","rating"])
        for i in range(1, n + 1):
            cat = random.choice(CATEGORIES)
            w.writerow([
                i,
                f"{cat.title()} Item {i}",
                cat,
                round(random.uniform(5.0, 500.0), 2),
                random.randint(0, 1000),
                round(random.uniform(1.0, 5.0), 1),
            ])
    print(f"    done in {time.time()-t:.1f}s")


def gen_events(n: int = 50_000):
    path = "data/events.jsonl"
    print(f"  Generating {n:,} events → {path}")
    t = time.time()
    with open(path, "w") as f:
        for i in range(1, n + 1):
            f.write(json.dumps({
                "event_id": i,
                "user_id": random.randint(1, 10_000),
                "event_type": random.choice(EVENT_TYPES),
                "page": f"/page/{random.randint(1, 100)}",
                "session_id": f"sess_{random.randint(1, 20_000)}",
                "duration_ms": random.randint(100, 30_000),
                "timestamp": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z",
            }) + "\n")
    print(f"    done in {time.time()-t:.1f}s")


if __name__ == "__main__":
    print("Generating sample datasets...")
    gen_orders()
    gen_customers()
    gen_products()
    gen_events()
    print("\nAll datasets ready in data/")
    print("Run: python cli.py --data data/")
