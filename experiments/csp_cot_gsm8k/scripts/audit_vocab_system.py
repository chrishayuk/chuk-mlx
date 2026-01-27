#!/usr/bin/env python3
"""
Audit the vocabulary and domain system for consistency and completeness.
"""

import json
from pathlib import Path
from collections import defaultdict

VOCAB_DIR = Path("/Users/christopherhay/chris-source/chuk-ai/virtual-experts/packages/chuk-virtual-expert-arithmetic/src/chuk_virtual_expert_arithmetic/vocab")
SCHEMA_DIR = Path("/Users/christopherhay/chris-source/chuk-ai/virtual-experts/packages/chuk-virtual-expert-arithmetic/src/chuk_virtual_expert_arithmetic/schemas")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def audit_shared_vocab():
    """Audit shared vocabulary files."""
    print("\n" + "=" * 70)
    print("SHARED VOCABULARY AUDIT")
    print("=" * 70)

    shared_files = list(VOCAB_DIR.glob("*.json"))
    print(f"\nFound {len(shared_files)} shared vocab files:")

    total_items = 0
    for f in sorted(shared_files):
        data = load_json(f)
        if isinstance(data, dict):
            keys = list(data.keys())
            item_count = sum(len(v) if isinstance(v, list) else 1 for v in data.values())
            print(f"  {f.stem:15} - {len(keys)} keys, ~{item_count} items")
            print(f"                   Keys: {keys[:5]}{'...' if len(keys) > 5 else ''}")
            total_items += item_count
        elif isinstance(data, list):
            print(f"  {f.stem:15} - {len(data)} items (list)")
            total_items += len(data)

    print(f"\n  Total: ~{total_items} vocabulary items")
    return shared_files


def audit_domains():
    """Audit domain files for consistency."""
    print("\n" + "=" * 70)
    print("DOMAIN AUDIT")
    print("=" * 70)

    domains_dir = VOCAB_DIR / "domains"
    domain_files = list(domains_dir.glob("*.json"))
    print(f"\nFound {len(domain_files)} domains:")

    issues = []
    stats = defaultdict(list)

    for f in sorted(domain_files):
        data = load_json(f)
        name = data.get("name", f.stem)

        # Check required fields
        has_agent_templates = "agent_templates" in data
        has_old_agents = "agents" in data
        has_items = "items" in data
        has_verbs = "verbs" in data
        has_time_units = "time_units" in data

        # Check agent template structure
        agent_types = data.get("agent_types", [])
        templates = data.get("agent_templates", {})

        status = "✓" if has_agent_templates and not has_old_agents else "⚠"
        if has_old_agents:
            issues.append(f"{name}: Still has old 'agents' field (hardcoded names)")
            status = "✗"
        if not has_agent_templates:
            issues.append(f"{name}: Missing 'agent_templates' field")
            status = "✗"

        # Check for hardcoded names in templates
        for tmpl_name, tmpl in templates.items():
            if "source" not in tmpl and "letters" not in tmpl and "numbers" not in tmpl:
                issues.append(f"{name}.{tmpl_name}: Template missing source/letters/numbers")

        # Collect stats
        stats["items"].append((name, len(data.get("items", []))))
        stats["time_units"].append((name, len(data.get("time_units", []))))
        stats["agent_types"].append((name, len(agent_types)))

        print(f"\n  {status} {name}")
        print(f"      agent_types: {agent_types}")
        print(f"      templates: {list(templates.keys())}")
        print(f"      items: {data.get('items', [])[:4]}{'...' if len(data.get('items', [])) > 4 else ''}")
        print(f"      verbs: {data.get('verbs', {})}")
        print(f"      time_units: {[t.get('singular') for t in data.get('time_units', [])]}")

    if issues:
        print(f"\n  ⚠ ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  ✓ All domains pass validation")

    return domain_files, issues


def audit_schemas():
    """Audit schema files."""
    print("\n" + "=" * 70)
    print("SCHEMA AUDIT")
    print("=" * 70)

    categories = defaultdict(list)
    all_schemas = []

    for subdir in sorted(SCHEMA_DIR.iterdir()):
        if subdir.is_dir():
            for schema_file in sorted(subdir.glob("*.json")):
                data = load_json(schema_file)
                name = data.get("name", schema_file.stem)
                categories[subdir.name].append(name)
                all_schemas.append((subdir.name, name, data))

    print(f"\nFound {len(all_schemas)} schemas in {len(categories)} categories:")

    for cat, schemas in sorted(categories.items()):
        print(f"\n  {cat}/ ({len(schemas)} schemas)")
        for s in schemas[:5]:
            print(f"    - {s}")
        if len(schemas) > 5:
            print(f"    ... and {len(schemas) - 5} more")

    # Check schema structure
    print("\n  Schema Variable Analysis:")
    var_patterns = defaultdict(list)
    for cat, name, data in all_schemas:
        vars = list(data.get("variables", {}).keys())
        var_key = tuple(sorted(vars))
        var_patterns[var_key].append(name)

    print(f"    Found {len(var_patterns)} unique variable patterns:")
    for vars, schemas in sorted(var_patterns.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"      {list(vars)}: {len(schemas)} schemas")

    return all_schemas


def audit_domain_schema_compatibility():
    """Check which schemas work with which domains."""
    print("\n" + "=" * 70)
    print("DOMAIN-SCHEMA COMPATIBILITY")
    print("=" * 70)

    # Rate-based schemas need: agent1, agent2, item, time_unit, rate variables
    # Shopping schemas need: items with prices
    # Entity schemas need: person names, countable items

    rate_vars = {"rate1", "rate2", "time"}
    shopping_vars = {"price", "cost", "start"}

    domains_dir = VOCAB_DIR / "domains"
    domains = {}
    for f in domains_dir.glob("*.json"):
        data = load_json(f)
        name = data.get("name", f.stem)
        domains[name] = data

    print(f"\n  Domain Capabilities:")
    for name, domain in sorted(domains.items()):
        has_person = any(
            "source" in t and "names" in t.get("source", "")
            for t in domain.get("agent_templates", {}).values()
        )
        has_equipment = any(
            "letters" in t or "numbers" in t
            for t in domain.get("agent_templates", {}).values()
        )
        has_priced = "priced_items" in domain

        caps = []
        if has_person:
            caps.append("person")
        if has_equipment:
            caps.append("equipment")
        if has_priced:
            caps.append("priced")

        print(f"    {name:15} - {', '.join(caps)}")


def audit_agent_name_diversity():
    """Check diversity of potential agent names."""
    print("\n" + "=" * 70)
    print("AGENT NAME DIVERSITY")
    print("=" * 70)

    # Load names
    names_file = VOCAB_DIR / "names.json"
    names = load_json(names_file)

    print(f"\n  Shared Names:")
    print(f"    people:  {len(names.get('people', []))} names")
    print(f"    male:    {len(names.get('male', []))} names")
    print(f"    female:  {len(names.get('female', []))} names")
    print(f"    neutral: {len(names.get('neutral', []))} names")

    # Count pattern-based possibilities per domain
    domains_dir = VOCAB_DIR / "domains"
    print(f"\n  Pattern-based Agent Possibilities:")

    for f in sorted(domains_dir.glob("*.json")):
        data = load_json(f)
        name = data.get("name", f.stem)
        templates = data.get("agent_templates", {})

        possibilities = 0
        details = []
        for tmpl_name, tmpl in templates.items():
            if "source" in tmpl:
                # Reference to shared vocab
                source = tmpl["source"]
                if source == "names.people":
                    count = len(names.get("people", []))
                else:
                    count = 100  # estimate
                possibilities += count
                details.append(f"{tmpl_name}={count}")
            elif "letters" in tmpl:
                count = len(tmpl["letters"])
                possibilities += count
                details.append(f"{tmpl_name}={count}")
            elif "numbers" in tmpl:
                count = len(tmpl["numbers"])
                possibilities += count
                details.append(f"{tmpl_name}={count}")

        print(f"    {name:15} - {possibilities:4} options ({', '.join(details)})")


def main():
    print("=" * 70)
    print("VOCABULARY & DOMAIN SYSTEM AUDIT")
    print("=" * 70)

    shared_files = audit_shared_vocab()
    domain_files, domain_issues = audit_domains()
    schemas = audit_schemas()
    audit_domain_schema_compatibility()
    audit_agent_name_diversity()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Shared Vocab Files: {len(shared_files)}")
    print(f"  Domains: {len(domain_files)}")
    print(f"  Schemas: {len(schemas)}")
    print(f"  Domain Issues: {len(domain_issues)}")

    if domain_issues:
        print("\n  ⚠ Action Required: Fix domain issues listed above")
    else:
        print("\n  ✓ System is consistent and ready for use")


if __name__ == "__main__":
    main()
