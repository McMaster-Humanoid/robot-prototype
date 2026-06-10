# Documentation Index

**Project**: VLA Manipulation for Unitree G1 using LeRobot + Genesis
**Target**: Feb 8, 2026 Demo

---

## 📋 Quick Start (Read These First)

### 1. **FEB8_SUCCESS_PLAN.md** ⭐ START HERE
Complete 14-day plan for Feb 8 demo with week-by-week breakdown.
- Timeline and milestones
- Technical setup
- Demo structure
- Risk mitigation
- Success metrics

### 2. **vla_setup_short.md** ⭐ TECHNICAL REFERENCE
Concise technical guide (60 lines) with essential commands and setup.
- Installation steps
- Common issues & fixes
- Training commands
- Quick reference

---

## 📚 Detailed Documentation

### Setup & Installation

**vla_manipulation_setup.md** (Full detailed guide)
- Complete setup process
- LeRobot installation
- PushT training walkthrough
- All errors encountered and solutions
- Comprehensive troubleshooting

### Genesis & G1 Integration

**FEB8_DEMO_GENESIS.md**
- Genesis + LeRobot strategy
- Why Genesis vs MuJoCo
- G1 mesh file solution
- Implementation approach
- Fast track to working demo

**g1_genesis_setup.md**
- Genesis environment creation
- G1 URDF integration
- Camera setup
- Gymnasium interface
- LeRobot integration

### Status & Planning

**FEB8_SUCCESS_PLAN.md** (Current plan)
- 14-day timeline
- Week-by-week tasks
- Demo structure
- Backup plans
- Commands cheat sheet

**G1_GENESIS_STATUS.md**
- Current blockers resolved
- What works now
- Mesh file status
- Options and paths forward

**LEAD_RESPONSE_OPTIONS.md**
- Decision matrix based on resource availability
- Three paths forward
- Timeline estimates

---

## 🗂️ Historical Documentation (Reference Only)

### Superseded Plans
These were useful during development but current plan is in FEB8_SUCCESS_PLAN.md:

**CORRECTED_NEXT_STEPS.md**
- Original Genesis integration plan
- Environment template code
- Pre-mesh-files approach

**FEB8_DEMO_PLAN.md**
- Initial 14-day plan (MuJoCo approach)
- Superseded by FEB8_DEMO_GENESIS.md

**g1_setup_status.md**
- MuJoCo simulator attempt
- Unitree SDK blocker
- Historical context

**NEXT_STEPS.md**
- Very early planning document
- Pre-G1 integration

---

## 📖 How to Use This Documentation

### If you're just starting:
1. Read: **FEB8_SUCCESS_PLAN.md** (the master plan)
2. Skim: **vla_setup_short.md** (quick technical reference)
3. Reference: **vla_manipulation_setup.md** (when you hit issues)

### If you're implementing:
- **FEB8_SUCCESS_PLAN.md** - Your daily guide
- **vla_setup_short.md** - Command reference
- **G1_GENESIS_STATUS.md** - Technical status

### If you're debugging:
- **vla_manipulation_setup.md** - All errors and solutions documented
- **g1_genesis_setup.md** - Genesis-specific issues

### If you're presenting (Feb 8):
- **FEB8_SUCCESS_PLAN.md** - Demo structure and talking points
- Use the "Key Messages" section
- Reference the backup plan

---

## 🎯 Current Status (As of Feb 2, 2026)

```
✅ LeRobot 0.4.3 installed
✅ Genesis 0.3.13 working
✅ G1 meshes in place (64 STL files)
✅ G1 loads in Genesis successfully
✅ ACT training proven (PushT: 94% loss reduction)
✅ G1ManipulationEnv created
✅ RTX 4070 GPU available
🔄 Environment final testing
```

**Next milestone**: Record demonstrations (Day 3-4)

---

## 📁 File Locations

### Documentation
```
docs/
├── README.md                      # This file (index)
├── FEB8_SUCCESS_PLAN.md          # ⭐ Master plan (CURRENT)
├── vla_setup_short.md            # ⭐ Quick reference
└── FEB8_DEMO_GENESIS.md          # Genesis strategy

g1_manipulation/
└── README.md                      # Environment documentation

Root:
└── README.md                      # Project README
```

### Code
```
g1_manipulation/              # G1 environment (main code)
├── env.py                   # G1ManipulationEnv
├── test_env.py              # Test script
├── __init__.py
└── README.md                # Environment docs

feb8_demo/                   # Demo scripts
├── test_g1_quick.py        # Quick test (no viewer)
├── test_g1_with_viewer.py  # With visual (slow)
└── test_g1_no_visual.py    # Initial test

rl-training-genesis/         # G1 model & meshes
├── models/
│   ├── g1_12dof.urdf       # G1 robot model
│   └── meshes/             # 64 STL files ✅
├── go2_env.py              # Reference (locomotion)
├── go2_eval.py
└── go2_train.py
```

### Outputs
```
outputs/
└── act_pusht_run1/          # PushT training results
    └── checkpoints/
        └── 005000/          # Trained ACT model
```

---

## 🔧 Common Commands

### Testing
```bash
# Quick test (no viewer - FAST)
python feb8_demo/test_g1_quick.py

# With viewer (SLOW in WSL2)
python feb8_demo/test_g1_with_viewer.py
```

### Training
```bash
# Train ACT on dataset
lerobot-train \
  --policy.type act \
  --dataset.repo_id your-username/g1-reach \
  --steps 5000 \
  --output_dir outputs/g1_feb8_demo \
  --policy.push_to_hub false \
  --dataset.video_backend pyav
```

---

## 🆘 If You Need Help

1. **Technical issue**: Check `vla_manipulation_setup.md` (all errors documented)
2. **Planning question**: See `FEB8_SUCCESS_PLAN.md` (timeline & approach)
3. **Genesis issue**: See `g1_genesis_setup.md` (G1-specific)
4. **Quick command**: See `vla_setup_short.md` (reference)

---

## 📝 Note on Historical Docs

Files marked "Historical" in this index are kept for reference but superseded by newer plans. The current active plan is always **FEB8_SUCCESS_PLAN.md**.

**Don't delete historical docs** - they contain useful context and may be referenced if you need to understand earlier decisions.

---

**Last Updated**: Feb 2, 2026
**Next Review**: After environment testing complete
**Owner**: McMaster Humanoid Team
