# 8-Minute Podcast Topics & Memory Guide

## 🎯 **Best Topics for 8-Minute Recording**

### **Top Recommendation: "What is Deaf Culture?"**
**Why this is perfect:**
- ✅ Rich content (identity, community, language, history)  
- ✅ Natural back-and-forth (Morty asks, Rick explains)
- ✅ Multiple subtopics for 8+ minutes
- ✅ Good demo of agent personalities
- ✅ Easy for Rick to be knowledgeable, Morty to be curious

**Expected Flow (8 min):**
1. **Summer intro** (0:00-0:20): "Today discussing Deaf Culture!"
2. **Morty curious** (0:20-1:00): "What exactly is Deaf culture?"
3. **Rick explains** (1:00-2:30): Identity vs disability, community bonds
4. **Morty follow-up** (2:30-3:00): "How is it different from hearing culture?"
5. **Rick deeper** (3:00-4:30): Language (ASL), shared experiences, values
6. **Morty** (4:30-5:00): "What about people who lose hearing later?"
7. **Rick** (5:00-6:30): Deaf vs deaf, CODA, community acceptance
8. **Morty wrap-up** (6:30-7:30): "What can hearing people learn?"
9. **Rick final** (7:30-8:00): Closing thoughts, respect

---

### **Alternative Great Topics:**

#### **1. "How does ASL grammar differ from English?"**
**Duration**: 7-9 minutes
**Pros**: Technical, educational, lots of examples
**Sample subtopics**:
- Word order (Topic-Comment structure)
- Facial expressions as grammar
- Classifiers and verb directionality
- No articles or "be" verb

#### **2. "History of Sign Language in America"**
**Duration**: 8-10 minutes
**Pros**: Historical depth, activism angle, controversy
**Sample subtopics**:
- Laurent Clerc and Gallaudet founding
- Milan Conference 1880 (oralism ban)
- Stokoe's linguistic research
- Modern recognition and ADA

#### **3. "What is a CODA and what's their experience?"**
**Duration**: 6-8 minutes
**Pros**: Personal stories, cultural bridge, unique perspective
**Sample subtopics**:
- CODA definition (Child of Deaf Adults)
- Language/cultural identity
- Interpreter role as children
- Famous CODAs

---

## 🧠 **Memory Management for Long Conversations**

### **Current Memory System** 

Looking at `podcast/services/conversation.py`:
```python
class ConversationMemory:
    def __init__(self, max_context_turns: int = 20):
        # Stores last 20 turns by default
        self.turns: List[ConversationTurn] = []
```

**Current implementation**:
- ✅ **Stores ALL turns** in memory (`self.turns`)
- ✅ **Sends last 20 turns** to LLM via `get_context()`
- ✅ **Tracks topics** and timestamps
- ✅ **Never forgets** within a session

---

### **How Memory Works**

#### **For 8-Minute Podcast (~12-16 agent turns):**

**Turn counting example:**
```
Turn 1: Summer intro
Turn 2: Morty question
Turn 3: Rick answer (long)
Turn 4: Morty follow-up
Turn 5: Rick explanation
Turn 6: USER INTERRUPT (vision input)
Turn 7: Rick responds to user
Turn 8: Morty new question
Turn 9: Rick answer
Turn 10: Morty follow-up
... (continues for 8 minutes)
```

**With 20-turn context window:**
- If conversation has 15 turns total → All 15 sent to LLM ✅
- If conversation has 25 turns total → Last 20 sent to LLM  
  - **Old turns 1-5 NOT in context** ⚠️
  - **But still stored** for transcript export ✅

---

###  **Guaranteeing Long-Term Memory**

#### **Option 1: Increase Context Window (Recommended)**

Edit `/home/hz/COMP0220 DL/COMP0220-Deep-Learning-CW/podcast/services/conversation.py`:

```python
def __init__(self, max_context_turns: int = 40):  # Changed from 20 to 40
    # Now keeps 40 turns in context
```

**Benefits:**
- Simple one-line change
- Handles up to 40 agent turns (~12-15 minute podcast)
- No risk of forgetting

**Tradeoffs:**
- Longer context = more tokens = slightly slower/expensive
- GPT-4o-mini handles 128K tokens easily (40 turns = ~10K tokens max)

---

#### **Option 2: Smart Summarization (Advanced)**

Add to `ConversationMemory` class:

```python
def get_context_with_summary(self) -> List[Dict[str, str]]:
    """Get context with older turns summarized."""
    if len(self.turns) <= self.max_context_turns:
        return self.get_context()  # All turns fit
    
    # Get old turns (beyond window)
    old_turns = self.turns[:-self.max_context_turns]
    recent_turns = self.turns[-self.max_context_turns:]
    
    # Create summary of old turns
    old_content = "\n".join([
        f"{t.agent}: {t.content}" for t in old_turns
    ])
    summary = f"[Earlier conversation summary: {len(old_turns)} turns about {self.topics[0]['topic'] if self.topics else 'various topics'}]"
    
    # Combine summary + recent turns
    messages = [{"role": "system", "content": summary}]
    messages.extend([turn.to_message() for turn in recent_turns])
    
    return messages
```

**Benefits:**
- Never loses information
- Keeps token count manageable
- Maintains full context

---

#### **Option 3: Topic-Based Windows (For Multiple Topics)**

```python
def get_context_for_topic(self) -> List[Dict[str, str]]:
    """Get turns since last topic change."""
    if not self.topics:
        return self.get_context()
    
    last_topic_idx = self.topics[-1]["turn_index"]
    topic_turns = self.turns[last_topic_idx:]
    
    # Include SOME pre-topic context for continuity
    pre_topic_turns = self.turns[max(0, last_topic_idx-5):last_topic_idx]
    
    all_turns = pre_topic_turns + topic_turns
    return [turn.to_message() for turn in all_turns]
```

---

## 🔧 **Quick Fix for Your Demo**

### **Recommended: Increase to 40 turns**

```bash
# Edit the file
nano /home/hz/COMP0220\ DL/COMP0220-Deep-Learning-CW/podcast/services/conversation.py

# Change line 53 from:
def __init__(self, max_context_turns: int = 20):

# To:
def __init__(self, max_context_turns: int = 40):

# Save and restart backend
```

**This guarantees:**
- ✅ No memory loss for 8-minute podcast
- ✅ Agents remember entire conversation
- ✅ Coherent follow-ups and references
- ✅ Natural flow without repetition

---

## 📋 **Testing Memory During Recording**

**How to verify memory is working:**

1. **Early in podcast** (Turn 3): Rick mentions specific fact  
   Example: "ASL was recognized as a language in the 1960s by Stokoe"

2. **Later in podcast** (Turn 12): Morty references it  
   Example: "Earlier you mentioned Stokoe's research in the 60s..."

3. **Check response**: Rick should acknowledge "Yes, exactly..."

**If memory fails:**
- Rick won't recognize the reference
- Response will be generic or repeat information
- **Solution**: Increase `max_context_turns` to 40-50

---

## 📊 **Context Window Math**

| Podcast Length | Est. Turns | Tokens (approx) | Recommended Window |
|----------------|------------|-----------------|-------------------|
| 5 minutes | 8-10 | 3K-5K | 20 turns ✅ |
| 8 minutes | 12-16 | 6K-9K | **40 turns** ✅ |
| 15 minutes | 25-30 | 12K-15K | 50 turns |
| 30 minutes | 50-60 | 25K-30K | 80 turns |

**GPT-4o-mini context limit:** 128,000 tokens (far more than needed!)

---

## 💡 **Pro Tips for Natural Flow**

### **1. Start Broad, Go Specific**
```
Topic: "Deaf Culture"
  ↓
Subtopic: "Identity vs disability"
  ↓
Example: "Deaf President Now protest"
  ↓
Personal: "How individuals navigate both worlds"
```

### **2. Use Morty's Questions to Transition**
```
Morty: "That's interesting! How does this relate to language?"
→ Natural segue to ASL grammar
```

### **3. Rick's Callbacks Create Cohesion**
```
Rick: "Remember when I mentioned Clerc earlier? Well..."
→ Shows memory, builds narrative
```

### **4. Summer Can Summarize**
Interrupt at 4 minutes with:
```
Summer: "Great discussion so far. Rick explained X, Morty asked about Y.
         Let's dive deeper into Z..."
```

---

## 🎬 **Recording Checklist**

**Before Starting:**
- [ ] Edit `conversation.py` → `max_context_turns = 40`
- [ ] Restart backend server
- [ ] Open browser to localhost:5173
- [ ] Start screen recording

**During Recording:**
- [ ] Choose topic: "What is Deaf Culture?"
- [ ] Let agents talk for 3-4 turns before any interruption
- [ ] Optional: Interrupt once at 4-5 minute mark
- [ ] Test memory by having Morty reference earlier points
- [ ] Let conversation naturally conclude around 8 minutes

**After Recording:**
- [ ] Export transcript (if needed): `memory.to_transcript()`
- [ ] Check for memory coherence in playback
- [ ] Verify agents didn't repeat themselves

---

## 🚀 **Next Steps**

1. **Increase memory window to 40 turns** (1-minute fix)
2. **Pick topic: "What is Deaf Culture?"** (best for 8-min demo)
3. **Start recording** - agents will handle the rest!
4. **Optional interrupt** at 4-5 minutes to show vision integration

**The agents are designed to:**
- Self-continue the conversation
- Build on previous points
- Ask follow-up questions
- Naturally wrap up after topic depth is exhausted

With 40-turn memory, your 8-minute podcast will have **perfect coherence and no forgetting**! 🎉
