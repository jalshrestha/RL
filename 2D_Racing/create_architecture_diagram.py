"""
Create system architecture diagram for CarRacing PPO implementation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
colors = {
    'env': '#4A90E2',      # Blue
    'wrapper': '#7ED321',  # Green
    'monitor': '#B8E986',  # Light green
    'vector': '#F5A623',   # Orange
    'stack': '#BD10E0',    # Purple
    'policy': '#D0021B',   # Red
}

# Define boxes with (x, y, width, height, label, color)
boxes = [
    (0.5, 2.5, 2.5, 1.0, 'CarRacing-v3\nEnvironment', colors['env']),
    (3.5, 2.5, 2.5, 1.0, 'Reward Shaping\nWrapper', colors['wrapper']),
    (6.5, 2.5, 2.0, 1.0, 'Monitor\n(Logging)', colors['monitor']),
    (9.0, 2.5, 2.5, 1.0, 'Parallel VecEnv\n(SubprocVecEnv\nor DummyVecEnv)', colors['vector']),
    (12.0, 2.5, 2.0, 1.0, 'Frame Stack\n(4 frames)', colors['stack']),
    (14.5, 2.5, 2.0, 1.0, 'PPO CNN\nPolicy', colors['policy']),
]

# Draw boxes
for x, y, w, h, label, color in boxes:
    # Create rounded rectangle
    fancy_box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        linewidth=2,
        edgecolor='black',
        facecolor=color,
        zorder=2
    )
    ax.add_patch(fancy_box)
    
    # Add text
    ax.text(x + w/2, y + h/2, label, 
            ha='center', va='center', 
            fontsize=10, fontweight='bold',
            color='white' if color in [colors['env'], colors['policy'], colors['stack']] else 'black')

# Draw arrows
arrows = [
    (3.0, 3.0, 3.5, 3.0),  
    (6.0, 3.0, 6.5, 3.0),  
    (8.5, 3.0, 9.0, 3.0),  
    (11.5, 3.0, 12.0, 3.0),  
    (14.0, 3.0, 14.5, 3.0),  
]

for x1, y1, x2, y2 in arrows:
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->',
        mutation_scale=30,
        linewidth=2.5,
        color='black',
        zorder=3
    )
    ax.add_patch(arrow)

# Add title
ax.text(7, 5.2, 'System Architecture', 
        ha='center', fontsize=16, fontweight='bold')

# Add details box
detail_text = """Flow: Environment → Reward Shaping → Monitor → 
Parallel Environments (8x) → Frame Stacking (4 frames) → PPO CNN Policy

Observations: 96×96×3 RGB images → 96×96×12 (after frame stacking)
Actions: Continuous [steering, gas, brake]"""
        
ax.text(7, 0.8, detail_text, 
        ha='center', fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('Image/System_Architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print(" Architecture diagram saved")

