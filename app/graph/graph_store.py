"""
Graph store for the GraphRAG Investigation Agent.

Provides a single interface (`GraphStore`) with two backends:
  - Neo4jGraphStore: real driver, used in staging/production against
    an actual Neo4j instance (set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD).
  - InMemoryGraphStore: NetworkX-backed, used for local dev/tests so the
    agent pipeline runs without any external infra.

Both implement: load_data(), find_fraud_rings(), get_customer_neighborhood().
"""
import os
import networkx as nx
from itertools import combinations


class InMemoryGraphStore:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def load_data(self, customers, accounts, transactions):
        for c in customers:
            self.g.add_node(c["customer_id"], type="Customer", **c)
        for a in accounts:
            self.g.add_node(a["account_id"], type="Account")
            self.g.add_edge(a["customer_id"], a["account_id"], rel="OWNS")

        for t in transactions:
            dev = t.get("device_id")
            ip = t.get("ip_address")
            if dev:
                self.g.add_node(dev, type="Device")
                self.g.add_edge(t["account_id"], dev, rel="USES")
            if ip:
                self.g.add_node(ip, type="IPAddress")
                self.g.add_edge(t["account_id"], ip, rel="USES")
            if t.get("transferred_to"):
                self.g.add_edge(t["account_id"], t["transferred_to"],
                                 rel="TRANSFERRED_TO", amount=t["amount"],
                                 timestamp=t["timestamp"])

    def find_fraud_rings(self, min_ring_size=3, max_shared_group_size=15,
                          min_ring_density=0.6):
        """
        Detect clusters of accounts that share a device or IP AND have
        transfer relationships tightly among that same group -- the
        classic mule-ring signature (shared infra + circular/chained
        money movement).

        Two guardrails against false positives on a small device/IP pool:
          - max_shared_group_size: if hundreds of unrelated accounts
            happen to share a device (pool exhaustion artifact, common
            in synthetic/sparse data), don't treat that as a ring.
          - min_ring_density: require most of the accounts sharing the
            entity to also participate in the transfer chain, not just
            one coincidental edge inside a big unrelated group.
        """
        rings = []
        device_ip_nodes = [n for n, d in self.g.nodes(data=True)
                            if d.get("type") in ("Device", "IPAddress")]

        for shared_node in device_ip_nodes:
            connected_accounts = list({
                u for u, v in self.g.in_edges(shared_node)
                if self.g.nodes[u].get("type") == "Account"
            })
            if len(connected_accounts) < min_ring_size or len(connected_accounts) > max_shared_group_size:
                continue

            connected_set = set(connected_accounts)
            transfer_edges = [
                (u, v, data) for u, v, data in self.g.edges(data=True)
                if data.get("rel") == "TRANSFERRED_TO" and u in connected_set and v in connected_set
            ]
            if not transfer_edges:
                continue

            participants = {u for u, v, _ in transfer_edges} | {v for u, v, _ in transfer_edges}
            density = len(participants) / len(connected_accounts)
            if density < min_ring_density:
                continue

            rings.append({
                "shared_entity": shared_node,
                "accounts": connected_accounts,
                "ring_participants": sorted(participants),
                "transfer_chain": transfer_edges,
                "risk_score": min(0.99, 0.6 + 0.08 * len(participants)),
            })

        return rings

    def get_customer_neighborhood(self, customer_id, depth=2):
        if customer_id not in self.g:
            return {"nodes": [], "edges": []}
        nodes = {customer_id}
        frontier = {customer_id}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                next_frontier |= set(self.g.successors(n)) | set(self.g.predecessors(n))
            nodes |= next_frontier
            frontier = next_frontier

        sub = self.g.subgraph(nodes)
        return {
            "nodes": [{"id": n, **sub.nodes[n]} for n in sub.nodes],
            "edges": [{"from": u, "to": v, **d} for u, v, d in sub.edges(data=True)],
        }


class Neo4jGraphStore:
    """Production backend. Requires `neo4j` driver and a running instance."""

    def __init__(self, uri=None, user=None, password=None):
        from neo4j import GraphDatabase
        uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.environ.get("NEO4J_USER", "neo4j")
        password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load_data(self, customers, accounts, transactions):
        with self.driver.session() as session:
            for c in customers:
                session.run(
                    "MERGE (c:Customer {customer_id: $id}) "
                    "SET c += $props",
                    id=c["customer_id"], props=c,
                )
            for a in accounts:
                session.run(
                    "MERGE (acc:Account {account_id: $id}) "
                    "WITH acc MATCH (c:Customer {customer_id: $cust_id}) "
                    "MERGE (c)-[:OWNS]->(acc)",
                    id=a["account_id"], cust_id=a["customer_id"],
                )
            for t in transactions:
                if t.get("device_id"):
                    session.run(
                        "MERGE (d:Device {device_id: $dev}) "
                        "WITH d MATCH (acc:Account {account_id: $acc}) "
                        "MERGE (acc)-[:USES]->(d)",
                        dev=t["device_id"], acc=t["account_id"],
                    )
                if t.get("transferred_to"):
                    session.run(
                        "MATCH (a:Account {account_id: $from_acc}) "
                        "MATCH (b:Account {account_id: $to_acc}) "
                        "MERGE (a)-[r:TRANSFERRED_TO {amount: $amt, timestamp: $ts}]->(b)",
                        from_acc=t["account_id"], to_acc=t["transferred_to"],
                        amt=t["amount"], ts=t["timestamp"],
                    )

    def find_fraud_rings(self, min_ring_size=3, max_shared_group_size=15,
                          min_ring_density=0.6):
        """
        Mirrors InMemoryGraphStore.find_fraud_rings()'s logic exactly, so
        callers get identical behavior regardless of backend: require
        actual TRANSFERRED_TO edges among the device/IP-sharing group
        (not just coincidental shared infrastructure), with the same
        false-positive guardrails (group size cap, density threshold).
        Returns plain strings/primitives only -- never raw Neo4j Node
        objects -- so results are JSON-serializable and match the shape
        the agent pipeline and API expect. Plain Cypher only (no APOC
        dependency), since APOC isn't guaranteed available on every
        managed Neo4j tier (e.g. some AuraDB Free configurations).
        """
        query = """
        MATCH (shared)
        WHERE shared:Device OR shared:IPAddress
        MATCH (acct:Account)-[:USES]->(shared)
        WITH shared, collect(DISTINCT acct.account_id) AS connected_accounts
        WHERE size(connected_accounts) >= $min_size AND size(connected_accounts) <= $max_group_size
        MATCH (a:Account)-[t:TRANSFERRED_TO]->(b:Account)
        WHERE a.account_id IN connected_accounts AND b.account_id IN connected_accounts
        WITH shared, connected_accounts,
             collect(DISTINCT a.account_id) + collect(DISTINCT b.account_id) AS participants_raw,
             collect({from: a.account_id, to: b.account_id, amount: t.amount, timestamp: t.timestamp}) AS transfer_chain
        UNWIND participants_raw AS p
        WITH shared, connected_accounts, collect(DISTINCT p) AS participants, transfer_chain
        WHERE size(participants) * 1.0 / size(connected_accounts) >= $min_density
        RETURN
            CASE WHEN 'Device' IN labels(shared) THEN shared.device_id ELSE shared.ip_address END AS shared_entity,
            connected_accounts AS accounts,
            participants AS ring_participants,
            transfer_chain,
            (0.6 + 0.08 * size(participants)) AS risk_score
        """
        params = {
            "min_size": min_ring_size,
            "max_group_size": max_shared_group_size,
            "min_density": min_ring_density,
        }
        with self.driver.session() as session:
            result = session.run(query, **params)
            records = [record.data() for record in result]
            for r in records:
                r["risk_score"] = min(0.99, r["risk_score"])
            return records

    def get_customer_neighborhood(self, customer_id, depth=2):
        """
        Returns plain dicts (matching InMemoryGraphStore's shape:
        {"nodes": [...], "edges": [...]}), not raw Neo4j Path objects --
        so this is JSON-serializable and consistent across both backends.
        """
        query = f"""
        MATCH path = (c:Customer {{customer_id: $cid}})-[*1..{depth}]-(n)
        RETURN path
        """
        nodes_by_id = {}
        edges = []
        with self.driver.session() as session:
            result = session.run(query, cid=customer_id)
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    node_id = (node.get("customer_id") or node.get("account_id")
                               or node.get("device_id") or node.get("ip_address") or node.element_id)
                    nodes_by_id[node_id] = {"id": node_id, "labels": list(node.labels), **dict(node)}
                for rel in path.relationships:
                    edges.append({
                        "from": dict(rel.start_node).get("account_id") or dict(rel.start_node).get("customer_id"),
                        "to": dict(rel.end_node).get("account_id") or dict(rel.end_node).get("customer_id"),
                        "rel": rel.type,
                        **dict(rel),
                    })
        return {"nodes": list(nodes_by_id.values()), "edges": edges}


def get_graph_store():
    """Factory: use Neo4j if configured, otherwise in-memory."""
    if os.environ.get("NEO4J_URI"):
        try:
            return Neo4jGraphStore()
        except Exception:
            pass
    return InMemoryGraphStore()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.core.synthetic_data import generate_dataset

    ds = generate_dataset()
    store = InMemoryGraphStore()
    store.load_data(ds["customers"], ds["accounts"], ds["transactions"])
    rings = store.find_fraud_rings()
    print(f"Found {len(rings)} fraud ring(s)")
    for r in rings:
        print(f"  shared_entity={r['shared_entity']} accounts={r['accounts']} risk={r['risk_score']:.2f}")
