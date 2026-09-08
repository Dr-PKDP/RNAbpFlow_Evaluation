"""
Resolves the author-assigned chain ID (`auth_asym_id`) for each clean
candidate. This differs from the `label_asym_id` typically returned by
RCSB metadata queries; legacy PDB-format files and standard structure
parsers key on the author ID, so it must be resolved explicitly rather
than assumed to match.

Candidates with multiple `auth_asym_id` values under a single RNA
polymer entity are resolved automatically (these are physically
duplicate copies of the same molecule in one asymmetric unit, a common
crystallographic outcome -- any one copy is equivalent for featurization
purposes). Candidates with genuinely multiple distinct RNA entities are
flagged for manual review rather than resolved automatically.

Usage:
    python fetch_auth_chains.py <clean_candidate_ids.json>

Writes auth_chain_map.json and, if applicable, multi_chain_review.json.
"""
import json
import sys
import time
import urllib.request

GRAPHQL_URL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 50


def fetch_auth_chains(pdb_ids: list[str]):
    query = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        polymer_entities {
          entity_poly { rcsb_entity_polymer_type }
          rcsb_polymer_entity_container_identifiers { auth_asym_ids }
        }
      }
    }
    """
    body = json.dumps({"query": query, "variables": {"ids": pdb_ids}}).encode()
    req = urllib.request.Request(GRAPHQL_URL, data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    result = {}
    multi_chain = []
    no_chain = []
    for entry in data.get("data", {}).get("entries", []) or []:
        rcsb_id = entry["rcsb_id"]
        rna_entities = []
        for pe in entry.get("polymer_entities", []) or []:
            if (pe.get("entity_poly") or {}).get("rcsb_entity_polymer_type") == "RNA":
                ids = (pe.get("rcsb_polymer_entity_container_identifiers") or {})
                auth_chains = ids.get("auth_asym_ids") or []
                if auth_chains:
                    rna_entities.append(auth_chains)

        if len(rna_entities) == 0:
            no_chain.append(rcsb_id)
        elif len(rna_entities) == 1:
            result[rcsb_id] = rna_entities[0][0]
        else:
            multi_chain.append((rcsb_id, rna_entities))
    return result, multi_chain, no_chain


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        clean_ids = json.load(f)
    print(f"Resolving author chain IDs for {len(clean_ids)} candidates")

    chain_map = {}
    all_multi = []
    all_none = []
    for i in range(0, len(clean_ids), BATCH_SIZE):
        batch = clean_ids[i:i + BATCH_SIZE]
        try:
            result, multi_chain, no_chain = fetch_auth_chains(batch)
        except Exception as exc:
            print(f"  batch {i}-{i+len(batch)}: request failed ({exc})")
            continue
        chain_map.update(result)
        all_multi.extend(multi_chain)
        all_none.extend(no_chain)
        print(f"  processed {min(i+BATCH_SIZE, len(clean_ids))}/{len(clean_ids)}")
        time.sleep(0.2)

    print(f"\nResolved: {len(chain_map)}")
    print(f"Multiple RNA entities (needs review): {len(all_multi)}")
    print(f"No RNA chain found: {len(all_none)}")

    with open("auth_chain_map.json", "w") as f:
        json.dump(chain_map, f, indent=2)
    if all_multi:
        with open("multi_chain_review.json", "w") as f:
            json.dump(all_multi, f, indent=2)
    if all_none:
        with open("no_chain_review.json", "w") as f:
            json.dump(all_none, f, indent=2)


if __name__ == "__main__":
    main()
