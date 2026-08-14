"""
Synthetic data generator for local development and testing.
Produces realistic-ish customers, accounts, and transactions,
including a deliberately injected fraud ring so the GraphRAG
agent has something real to find.
"""
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

MERCHANTS = ["Amazon", "Walmart", "Shell", "Uber", "BestBuy", "Target", "Costco", "Apple"]
COUNTRIES = ["US", "US", "US", "CA", "GB", "NG", "RU"]
CITIES = ["New York", "Chicago", "Miami", "Dallas", "Seattle"]


def _cust_id(i):
    return f"CUST{i:05d}"


def _acct_id(i):
    return f"ACC{i:05d}"


def generate_customers(n=200):
    customers = []
    for i in range(n):
        customers.append({
            "customer_id": _cust_id(i),
            "name": f"Customer {i}",
            "kyc_status": random.choice(["VERIFIED", "VERIFIED", "VERIFIED", "PENDING"]),
            "risk_level": random.choice(["LOW", "LOW", "MEDIUM", "HIGH"]),
            "country": random.choice(COUNTRIES),
        })
    return customers


def generate_accounts(customers):
    accounts = []
    for c in customers:
        for _ in range(random.randint(1, 2)):
            accounts.append({
                "account_id": _acct_id(len(accounts)),
                "customer_id": c["customer_id"],
            })
    return accounts


def generate_transactions(customers, accounts, n=2000, inject_fraud_ring=True):
    txns = []
    now = datetime.utcnow()
    acct_ids = [a["account_id"] for a in accounts]
    acct_to_cust = {a["account_id"]: a["customer_id"] for a in accounts}

    devices = [f"DEV{i:04d}" for i in range(50)]
    ips = [f"192.168.{i}.{j}" for i in range(1, 6) for j in range(1, 20)]

    for i in range(n):
        acct = random.choice(acct_ids)
        amount = round(random.uniform(5, 500), 2)
        is_anomalous = random.random() < 0.03
        if is_anomalous:
            amount = round(random.uniform(2000, 15000), 2)

        txns.append({
            "transaction_id": f"TX{i:06d}",
            "account_id": acct,
            "customer_id": acct_to_cust[acct],
            "amount": amount,
            "merchant": random.choice(MERCHANTS),
            "timestamp": (now - timedelta(minutes=random.randint(0, 60 * 24 * 30))).isoformat(),
            "device_id": random.choice(devices),
            "ip_address": random.choice(ips),
            "location": random.choice(CITIES),
            "label_fraud": 1 if is_anomalous else 0,
        })

    if inject_fraud_ring:
        # Create a small ring: 5 accounts sharing 1 device + 1 IP,
        # with rapid transfers between them (classic mule pattern).
        ring_accts = random.sample(acct_ids, 5)
        shared_device = "DEV_RING_001"
        shared_ip = "10.0.0.99"
        base_time = now - timedelta(hours=2)
        for j, acct in enumerate(ring_accts):
            txns.append({
                "transaction_id": f"TXRING{j:04d}",
                "account_id": acct,
                "customer_id": acct_to_cust[acct],
                "amount": round(random.uniform(3000, 9000), 2),
                "merchant": "WIRE_TRANSFER",
                "timestamp": (base_time + timedelta(minutes=j * 4)).isoformat(),
                "device_id": shared_device,
                "ip_address": shared_ip,
                "location": "Unknown",
                "label_fraud": 1,
                "transferred_to": ring_accts[(j + 1) % len(ring_accts)],
            })

    return txns


def generate_dataset(n_customers=200, n_transactions=2000):
    customers = generate_customers(n_customers)
    accounts = generate_accounts(customers)
    transactions = generate_transactions(customers, accounts, n_transactions)
    return {"customers": customers, "accounts": accounts, "transactions": transactions}


if __name__ == "__main__":
    import json
    ds = generate_dataset()
    print(f"customers={len(ds['customers'])} accounts={len(ds['accounts'])} txns={len(ds['transactions'])}")
    print(json.dumps(ds["transactions"][-5:], indent=2))
