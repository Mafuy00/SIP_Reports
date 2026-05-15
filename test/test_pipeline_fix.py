#!/usr/bin/env python3
"""
Quick test to verify the patched RAG pipeline works as expected:
- Retrieval should be fast
- Generation should fail fast with a clear error message (model not installed)
- No hanging/indefinite delays
"""

import sys
import time
import rag_assets.rag_pipeline as p

def test_retrieval():
    print("=" * 70)
    print("TEST 1: RETRIEVAL (should be fast)")
    print("=" * 70)
    
    q = "What are the common learning outcomes for students in their SIP?"
    start = time.time()
    
    try:
        context = p.get_context(q, k=5)
        elapsed = time.time() - start
        
        print(f"✓ Retrieval succeeded in {elapsed:.2f}s")
        print(f"  Context length: {len(context)} characters")
        print(f"  Preview: {context[:150].replace(chr(10), ' ')[:150]}...")
        return context
    except Exception as e:
        print(f"✗ Retrieval failed: {type(e).__name__}: {e}")
        return None

def test_generation(context):
    print("\n" + "=" * 70)
    print("TEST 2: GENERATION (should fail fast with clear error)")
    print("=" * 70)
    
    q = "What are the common learning outcomes for students in their SIP?"
    start = time.time()
    
    try:
        answer = p.generate_answer(
            q,
            context=context,
            context_k=5,
            ollama_model="llama3",
            timeout_seconds=10,
        )
        elapsed = time.time() - start
        print(f"✓ Generation succeeded in {elapsed:.2f}s: {answer[:100]}...")
    except RuntimeError as e:
        elapsed = time.time() - start
        print(f"✓ Generation failed as expected (model not installed) in {elapsed:.2f}s:")
        print(f"  Error: {str(e)[:200]}...")
    except TimeoutError as e:
        elapsed = time.time() - start
        print(f"✓ Generation timed out as expected in {elapsed:.2f}s:")
        print(f"  Error: {str(e)[:200]}...")
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Unexpected error in {elapsed:.2f}s:")
        print(f"  {type(e).__name__}: {str(e)[:200]}...")
        return False
    
    # Verify it didn't take forever
    if elapsed > 60:
        print(f"\n✗ ERROR: Generation took {elapsed:.2f}s (> 60s timeout). This is the bug!")
        return False
    
    print(f"\n✓ SUCCESS: Generation failed fast (within timeout).")
    return True

if __name__ == "__main__":
    context = test_retrieval()
    if context:
        success = test_generation(context)
        sys.exit(0 if success else 1)
    else:
        print("\nSkipping generation test because retrieval failed.")
        sys.exit(1)
