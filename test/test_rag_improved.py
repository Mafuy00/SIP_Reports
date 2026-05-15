"""
Test script for improved RAG pipeline
Query: List all software, programming languages, and technical tools mentioned in the SIP reports.
"""

import sys
from pathlib import Path

# Set up the path
rag_assets_dir = Path("SIPReports_ToYunLin/rag_assets")
sys.path.insert(0, str(rag_assets_dir.resolve().parent))

# Use the improved pipeline
from rag_assets.rag_pipeline_improved import get_context, generate_answer
import os

print("="*80)
print("RAG PIPELINE TEST: Improved vs Original")
print("="*80)

question = "List all software, programming languages, and technical tools mentioned in the SIP reports."

print(f"\nQUESTION: {question}\n")

# Test 1: Get context with improved pipeline (hybrid search)
print("\n" + "="*80)
print("RETRIEVED CONTEXT (Improved Pipeline with Hybrid Search)")
print("="*80)

context_improved = get_context(question, k=6, use_hybrid=True, alpha=0.6)
print(context_improved)

# Test 2: Try to generate answer using Ollama if available
print("\n" + "="*80)
print("ATTEMPTING ANSWER GENERATION")
print("="*80)

# Check if ollama is available
import shutil
ollama_available = shutil.which("ollama") is not None

if ollama_available:
    print("✓ Ollama found, attempting to generate answer...")
    try:
        answer = generate_answer(
            question,
            context_k=6,
            use_hybrid=True,
            timeout_seconds=60
        )
        print("\nGENERATED ANSWER:")
        print(answer)
    except Exception as e:
        print(f"\nGeneration failed: {type(e).__name__}: {str(e)[:200]}")
        print("\nThis is expected if:")
        print("  • Ollama service is not running")
        print("  • Required model is not installed locally")
        print("\nThe retrieved context above shows the relevant sections found by RAG.")
else:
    print("✗ Ollama not found on PATH")
    print("\nIf you want to test answer generation:")
    print("  1. Install Ollama from https://ollama.ai")
    print("  2. Run: ollama serve")
    print("  3. In another terminal: ollama pull phi3 (or your preferred model)")
    print("  4. Re-run this script")

# Analysis
print("\n" + "="*80)
print("DIAGNOSTICS & ANALYSIS")
print("="*80)

print("""
The improved pipeline uses HYBRID SEARCH which combines:
  • Dense semantic similarity (60% weight)
  • Keyword-based TF-IDF search (40% weight)

For the query about "software, programming languages, and technical tools":
  1. Dense search finds semantically similar content
  2. Keyword search finds explicit mentions of specific tools
  3. Combined results should include multiple relevant sections

Expected findings:
  ✓ Multiple sections from different parts of the report
  ✓ Mix of semantically similar and keyword-matching results
  ✓ Better coverage of diverse tool mentions
  ✗ No single "irrelevant" high-scoring chunk from wrong section
  ✗ No duplicate sections in top results
""")

# Check the retrieved context characteristics
from pathlib import Path
import json

rag_dir = Path("SIPReports_ToYunLin/rag_assets")
with open(rag_dir / "rag_chunk_metadata.json", encoding="utf-8") as f:
    all_metadata = json.load(f)

# Analyze the retrieved sections
print("\nRETRIEVED SECTIONS ANALYSIS:")
print("-" * 80)

# Parse the context to see which sections were retrieved
# The format is "[Source: file | Section: title | Score: 0.xxx]"
import re
section_matches = re.findall(r'\[Source: [^|]+ \| Section: ([^|]+) \| Score: ([\d.]+)\]', context_improved)

print(f"Total sections retrieved: {len(section_matches)}")
print("\nSections in retrieval order:")
for i, (section, score) in enumerate(section_matches, 1):
    print(f"  {i}. {section[:60]:60s} (Score: {score})")

# Check for duplicates
unique_sections = set(s for s, _ in section_matches)
if len(unique_sections) < len(section_matches):
    duplicates = len(section_matches) - len(unique_sections)
    print(f"\n✗ WARNING: {duplicates} duplicate section(s) found")
else:
    print(f"\n✓ GOOD: All {len(unique_sections)} sections are unique")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
To improve results further:
1. Run diagnostics notebook to understand exact issues
2. Use the improved pipeline for all queries (already implemented)
3. If needed, regenerate chunks without embedded section titles
4. Monitor retrieval quality with your specific use cases
""")
