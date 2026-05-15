"""Quick test of RAG retrieval for technical tools query"""
import sys
from pathlib import Path

# Minimal setup
rag_dir = Path("SIPReports_ToYunLin/rag_assets")
sys.path.insert(0, str(rag_dir.resolve().parent))

print("Loading RAG pipeline...")
from rag_assets.rag_pipeline_improved import get_context

print("✓ Pipeline loaded\n")

question = "List all software, programming languages, and technical tools mentioned in the SIP reports."

print("="*80)
print("QUESTION:")
print(question)
print("="*80)

print("\nRetrieving context with improved pipeline (hybrid search)...")
context = get_context(question, k=6, use_hybrid=True, alpha=0.6)

print("\nRETRIEVED CONTEXT:")
print("-"*80)
print(context)

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

# Parse sections
import re
sections = re.findall(r'Section: ([^|]+)', context)
print(f"\nSections retrieved ({len(sections)}):")
for i, s in enumerate(sections, 1):
    print(f"  {i}. {s.strip()[:70]}")

# Check if it's finding relevant content
if any(word in context.lower() for word in ['python', 'java', 'sql', 'chatbot', 'technology', 'tool', 'software', 'programming']):
    print("\n✓ SUCCESS: Retrieved context contains mentions of technical tools/languages")
else:
    print("\n✗ WARNING: No obvious technical tool mentions found")
    
print("\nThe retrieval appears to be working. If using Ollama:")
print("  ollama serve (in one terminal)")
print("  python -c \"from rag_assets.rag_pipeline_improved import generate_answer; ...")
