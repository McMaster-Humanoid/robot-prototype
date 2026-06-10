# Feb 8th Demo - Genesis + LeRobot Strategy

**Deadline**: February 8, 2026 (14 days)
**Approach**: Genesis simulation + LeRobot ACT training

---

## Why Genesis (Not MuJoCo)?

✅ **You already use Genesis** for G1 RL training
✅ **GPU-accelerated** (faster than MuJoCo)
✅ **You're familiar with it** (less learning curve)
✅ **Works with LeRobot** (any Gym env works)

LeRobot's MuJoCo mode is for **physical robot deployment**. For simulation/training, Genesis is perfect!

---

## The Plan

### Genesis G1 Environment (Custom)
```
Genesis Simulation
    ↓
Gymnasium Wrapper (your code)
    ↓
LeRobot Recording/Training (existing tools)
```

### What We Have Already
- ✅ Genesis installed
- ✅ G1 URDF (`rl-training-genesis/models/g1_12dof.urdf`)
- ✅ LeRobot working (PushT trained successfully)
- ⚠️ **BLOCKER**: G1 mesh files missing

---

## Immediate Solutions

### Option A: Get G1 Meshes (Best)
**Where to find**:
1. Your team's G1 package
2. Unitree official repository
3. Ask teammates who have complete G1 setup

**Once found**:
```bash
mkdir -p /home/sehaj/one-leg-robot/rl-training-genesis/models/meshes/
# Copy all STL files there
```

### Option B: Remove Visual Meshes (Quick Workaround)
Create a simplified G1 URDF without visual meshes for simulation:
- Keep collision geometry
- Remove/simplify visual elements
- Focus on manipulation, not pretty graphics

### Option C: Use Simplified Robot (Fastest - Demo Ready Today)
Instead of full G1, use simplified arm-only robot:
- Single 7-DOF arm
- Simple geometric shapes
- Just for VLA demonstration
- Show the **VLA pipeline**, not specific robot

---

## Recommended: Option C for Demo

**Why**: You need a working demo in 14 days. Show the VLA **methodology**, not the specific G1 hardware.

### Quick Win Strategy
1. **Today**: Create simple arm manipulation in Genesis
2. **Week 1**: Record demos, train ACT
3. **Week 2**: Polish demo, create presentation
4. **Parallel**: Get G1 meshes for future work

---

## What to Build (Next 3 Days)

### Day 1: Simple Genesis Arm Environment
```python
# simple_arm_env.py
import genesis as gs
import gymnasium as gym

class SimpleArmEnv(gym.Env):
    """7-DOF arm reaching task in Genesis"""
    
    def __init__(self):
        gs.init(backend=gs.cuda)
        self.scene = gs.Scene(...)
        
        # Simple 7-DOF arm (boxes/cylinders, no meshes)
        self.arm = self.create_simple_arm()
        self.target = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), ...)
        )
        self.camera = self.scene.add_camera(...)
        
    def step(self, action):
        # Control arm joints
        obs = self._get_obs()  # image + joint states
        reward = self._compute_reward()
        return obs, reward, done, info
```

### Day 2: Test with LeRobot
```bash
# Record demonstrations (manual/scripted)
lerobot-record \
  --env-path simple_arm_env.py \
  --env-name SimpleArmEnv \
  --episodes 30 \
  --repo-id your-username/arm-reach-demo

# Train ACT
lerobot-train \
  --policy.type act \
  --dataset.repo_id your-username/arm-reach-demo \
  --steps 3000
```

### Day 3: Demo Script
```python
# demo_for_feb8.py
# Load trained policy
# Run in Genesis
# Show reaching task working
```

---

## Feb 8th Demo Structure

### Part 1: VLA Introduction (5 min)
**Concept**:
- Visual-Language-Action models
- Learn from demonstrations (not RL rewards)
- Input: images → Output: actions

**Why VLAs?**:
- Easier than reward engineering
- Leverages human demonstrations
- Works for complex manipulation

### Part 2: Our Implementation (3 min)
**Tech stack**:
- Genesis (GPU physics simulation)
- LeRobot (Hugging Face toolkit)
- ACT policy (Action Chunking Transformer)

**Process**:
1. Create manipulation task in Genesis
2. Record demonstrations
3. Train ACT policy on demos
4. Deploy and evaluate

### Part 3: Live Demo (5 min)
**Show**:
1. Genesis simulation running
2. Trained policy controlling arm
3. Reaching/manipulation task succeeding
4. Training curves/metrics

**Talking points**:
- "This same approach scales to G1"
- "We demonstrated the VLA pipeline"
- "Next: full G1 with mesh files"

---

## Timeline (14 Days)

| Day | Task | Output |
|-----|------|--------|
| 1-2 | Simple arm environment | Working Genesis env |
| 3-4 | Record demonstrations | Dataset (30 episodes) |
| 5-6 | Train ACT policy | Trained model |
| 7-8 | Test and debug | Working demo |
| 9-10 | Create presentation | Slides ready |
| 11-12 | Practice demo | Polished presentation |
| 13 | Buffer/improvements | Final touches |
| 14 | **DEMO DAY** | 🎉 |

---

## Code Structure

```
/home/sehaj/one-leg-robot/
├── feb8_demo/
│   ├── simple_arm_env.py      # Genesis arm environment
│   ├── record_demos.py         # Teleoperation script
│   ├── train_policy.py         # ACT training
│   ├── run_demo.py             # Live demo script
│   └── README.md               # Setup instructions
├── docs/
│   ├── FEB8_DEMO_GENESIS.md   # This file
│   └── VLA_PRESENTATION.md    # Slides content
└── outputs/
    └── feb8_demo/              # Trained models
```

---

## Backup Plan

**If Genesis has issues**:
1. Show PushT demo (already working)
2. Explain VLA concepts
3. Show Genesis simulation screenshots
4. Discuss G1 integration plan

**If time is tight**:
1. Use PushT as main demo
2. Show Genesis arm in development
3. Focus on VLA methodology

---

## Key Message for Demo

**Don't focus on**: "We got G1 working perfectly"

**DO focus on**: "We demonstrated VLA pipeline that works for any robot"

**Narrative**:
- "VLAs enable robots to learn manipulation from demos"
- "We built pipeline using Genesis + LeRobot"
- "Demonstrated on arm reaching task"
- "Same approach applies to G1 humanoid"
- "Next steps: full G1 integration"

---

## What You're Actually Demonstrating

Not "perfect G1 manipulation" but:
1. ✅ Understanding of VLA concepts
2. ✅ Working simulation pipeline
3. ✅ Data collection process
4. ✅ ACT training (proven on PushT)
5. ✅ Integration approach for G1

**This is what they asked for**: "show everyone... VLAs and how to use them in a simulated environment"

---

## Next Action (RIGHT NOW)

Choose fastest path:

### Path A: Get G1 Meshes Today
If you can get meshes from team/resources:
- Use full G1 in Genesis
- More impressive visually
- More time-consuming

### Path B: Simple Arm Demo (Recommended)
- Create simple arm today
- Working demo in 3 days
- Lower risk, guaranteed success
- Still demonstrates VLA pipeline

**My recommendation**: Path B (simple arm) for guaranteed success, pursue G1 meshes in parallel for future work.

---

## Commands to Start

### Create Simple Arm Environment
```bash
cd ~/one-leg-robot
mkdir feb8_demo
cd feb8_demo

# Create simple_arm_env.py (I can help write this)
# Test it works
python simple_arm_env.py
```

### Once Working
```bash
# Record demos
lerobot-record --env-path feb8_demo/simple_arm_env.py ...

# Train
lerobot-train --policy.type act ...
```

---

**Bottom Line**: 
- ✅ Use Genesis (what you know)
- ✅ Use LeRobot (proven working)
- ✅ Simple arm demo (low risk)
- ✅ Show VLA pipeline (core goal)
- ⏳ Full G1 (parallel effort, not blocking)

**Ready to start?** I can create the simple arm environment now!
