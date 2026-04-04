"""
Terrain Generation for Quadruped Training
Creates progressively difficult terrain for curriculum learning
"""
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


class TerrainGenerator:
    """Generate various types of terrain heightmaps for quadruped locomotion training"""
    
    def __init__(self, terrain_size: Tuple[float, float] = (10.0, 10.0), resolution: float = 0.05):
        """
        Args:
            terrain_size: (width, height) in meters
            resolution: grid resolution in meters (smaller = more detailed)
        """
        self.terrain_size = terrain_size
        self.resolution = resolution
        self.grid_size = (
            int(terrain_size[0] / resolution),
            int(terrain_size[1] / resolution)
        )
        self.difficulty_level = 0  # 0-9, progressively harder
        
    def set_difficulty(self, level: int):
        """Set terrain difficulty level (0=flat, 9=very challenging)"""
        self.difficulty_level = max(0, min(9, level))
    
    def generate_flat(self) -> np.ndarray:
        """Generate perfectly flat terrain (difficulty 0)"""
        return np.zeros(self.grid_size, dtype=np.float32)
    
    def generate_random_hills(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random hills with difficulty-based amplitude and frequency
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Difficulty scales amplitude only — width stays small so hills are distinct
        amplitude = 0.02 * (self.difficulty_level + 1)  # 0.02m to 0.20m
        # Many small hills cover the full terrain (40 at easy, 88 at hardest)
        num_hills = max(40, 8 * (self.difficulty_level + 1))

        height_map = np.zeros(self.grid_size, dtype=np.float32)

        # Pre-build meshgrid once outside the loop
        x = np.arange(self.grid_size[0])
        y = np.arange(self.grid_size[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        for _ in range(num_hills):
            # Allow centers across the full grid (including near edges) for uniform coverage
            center_x = np.random.uniform(-0.05, 1.05) * self.grid_size[0]
            center_y = np.random.uniform(-0.05, 1.05) * self.grid_size[1]

            # Small, tight hills (sigma 2–8 grid cells) so peaks are clearly distinct
            width = np.random.uniform(2, 8)
            height = np.random.uniform(0.5, 1.0) * amplitude

            distance_sq = ((X - center_x)**2 + (Y - center_y)**2)
            hill = height * np.exp(-distance_sq / (2 * width**2))

            # Mostly additive; occasional shallow valley for variety
            if np.random.random() > 0.25:
                height_map += hill
            else:
                height_map -= hill * 0.4

        return height_map
    
    def generate_sine_waves(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate sinusoidal terrain (rolling hills)
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        amplitude = 0.015 * (self.difficulty_level + 1)  # 0.015m to 0.15m
        frequency = 0.02 * (1 + self.difficulty_level * 0.3)  # Varies with difficulty
        
        x = np.arange(self.grid_size[0])
        y = np.arange(self.grid_size[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Multiple sine waves at different angles
        num_waves = 2 + min(self.difficulty_level // 2, 3)
        height_map = np.zeros(self.grid_size, dtype=np.float32)
        
        for i in range(num_waves):
            angle = np.random.uniform(0, np.pi)
            phase = np.random.uniform(0, 2 * np.pi)
            wave_freq = frequency * np.random.uniform(0.5, 1.5)
            wave_amp = amplitude * np.random.uniform(0.6, 1.0)
            
            rotated = X * np.cos(angle) + Y * np.sin(angle)
            height_map += wave_amp * np.sin(rotated * wave_freq + phase)
        
        return height_map
    
    def generate_stepped_terrain(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate stepped terrain (stairs/platforms)
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        step_height = 0.01 * (self.difficulty_level + 1)  # 0.01m to 0.10m
        num_steps = 3 + self.difficulty_level // 2  # 3 to 7 steps
        
        height_map = np.zeros(self.grid_size, dtype=np.float32)
        
        # Vertical or horizontal steps
        if np.random.random() > 0.5:
            # Horizontal steps
            step_width = self.grid_size[0] // num_steps
            for i in range(num_steps):
                start = i * step_width
                end = (i + 1) * step_width
                height = (i + 1) * step_height * (1 if np.random.random() > 0.3 else -0.5)
                height_map[start:end, :] = height
        else:
            # Vertical steps
            step_width = self.grid_size[1] // num_steps
            for i in range(num_steps):
                start = i * step_width
                end = (i + 1) * step_width
                height = (i + 1) * step_height * (1 if np.random.random() > 0.3 else -0.5)
                height_map[:, start:end] = height
        
        return height_map
    
    def generate_rough_terrain(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate rough/rocky terrain using Perlin-like noise
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        amplitude = 0.01 * (self.difficulty_level + 1)
        
        # Start with random noise
        height_map = np.random.randn(*self.grid_size).astype(np.float32)
        
        # Apply Gaussian blur for smoothness (Perlin-like)
        # Higher difficulty = less smoothing = rougher terrain
        sigma = max(2, 10 - self.difficulty_level)
        height_map = ndimage.gaussian_filter(height_map, sigma=sigma)
        
        # Normalize and scale
        height_map = height_map - height_map.min()
        height_map = height_map / (height_map.max() + 1e-8)
        height_map = (height_map - 0.5) * 2 * amplitude  # Center around 0
        
        return height_map
    
    def generate_mixed_terrain(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate mixed terrain combining multiple types
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Randomly combine different terrain types
        terrain_types = []
        weights = []
        
        # Always include some base terrain
        if self.difficulty_level < 3:
            terrain_types.append(self.generate_random_hills(seed))
            weights.append(0.7)
            terrain_types.append(self.generate_sine_waves(seed + 1 if seed else None))
            weights.append(0.3)
        elif self.difficulty_level < 6:
            terrain_types.append(self.generate_random_hills(seed))
            weights.append(0.5)
            terrain_types.append(self.generate_rough_terrain(seed + 1 if seed else None))
            weights.append(0.3)
            terrain_types.append(self.generate_sine_waves(seed + 2 if seed else None))
            weights.append(0.2)
        else:
            # High difficulty: all terrain types
            terrain_types.append(self.generate_random_hills(seed))
            weights.append(0.35)
            terrain_types.append(self.generate_rough_terrain(seed + 1 if seed else None))
            weights.append(0.35)
            terrain_types.append(self.generate_stepped_terrain(seed + 2 if seed else None))
            weights.append(0.3)
        
        # Weighted combination
        height_map = np.zeros(self.grid_size, dtype=np.float32)
        for terrain, weight in zip(terrain_types, weights):
            height_map += terrain * weight
        
        return height_map
    
    def smooth_terrain(self, height_map: np.ndarray, iterations: int = 1) -> np.ndarray:
        """
        Smooth terrain to avoid simulation instabilities
        
        Args:
            height_map: Input height map
            iterations: Number of smoothing iterations
        """
        for _ in range(iterations):
            height_map = ndimage.gaussian_filter(height_map, sigma=1.0)
        return height_map
    
    def visualize_terrain(self, height_map: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize terrain heightmap
        
        Args:
            height_map: Height map to visualize
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 5))
            
            # 2D view
            ax1 = fig.add_subplot(121)
            im = ax1.imshow(height_map.T, cmap='terrain', origin='lower')
            ax1.set_title(f'Terrain Height Map (Difficulty {self.difficulty_level})')
            ax1.set_xlabel('X (grid)')
            ax1.set_ylabel('Y (grid)')
            plt.colorbar(im, ax=ax1, label='Height (m)')
            
            # 3D view
            ax2 = fig.add_subplot(122, projection='3d')
            X = np.arange(self.grid_size[0])
            Y = np.arange(self.grid_size[1])
            X, Y = np.meshgrid(X, Y)
            ax2.plot_surface(X, Y, height_map.T, cmap='terrain', alpha=0.8)
            ax2.set_title('3D View')
            ax2.set_xlabel('X (grid)')
            ax2.set_ylabel('Y (grid)')
            ax2.set_zlabel('Height (m)')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Terrain visualization saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            print("Matplotlib not available. Cannot visualize terrain.")


class CurriculumManager:
    """Manage progressive difficulty curriculum for terrain training"""
    
    def __init__(self, terrain_generator: TerrainGenerator, change_interval: int = 500):
        """
        Args:
            terrain_generator: TerrainGenerator instance
            change_interval: Number of iterations between difficulty increases
        """
        self.terrain_generator = terrain_generator
        self.change_interval = change_interval
        self.current_difficulty = 0
        self.max_difficulty = 9
        self.last_change_iteration = 0
    
    def should_change_terrain(self, current_iteration: int) -> bool:
        """Check if it's time to change terrain difficulty"""
        return current_iteration > 0 and current_iteration % self.change_interval == 0
    
    def update(self, current_iteration: int) -> bool:
        """
        Update difficulty level if needed
        
        Returns:
            True if difficulty was increased, False otherwise
        """
        if self.should_change_terrain(current_iteration):
            if self.current_difficulty < self.max_difficulty:
                self.current_difficulty += 1
                self.terrain_generator.set_difficulty(self.current_difficulty)
                self.last_change_iteration = current_iteration
                print(f"\n{'='*60}")
                print(f"CURRICULUM UPDATE - Iteration {current_iteration}")
                print(f"Increased terrain difficulty: {self.current_difficulty - 1} → {self.current_difficulty}")
                print(f"Next change at iteration: {current_iteration + self.change_interval}")
                print(f"{'='*60}\n")
                return True
        return False
    
    def get_status(self) -> dict:
        """Get current curriculum status"""
        return {
            "difficulty": self.current_difficulty,
            "max_difficulty": self.max_difficulty,
            "last_change": self.last_change_iteration,
            "next_change": self.last_change_iteration + self.change_interval,
            "progress": f"{self.current_difficulty}/{self.max_difficulty}"
        }


if __name__ == "__main__":
    # Demo usage
    print("Terrain Generator Demo")
    print("=" * 60)
    
    generator = TerrainGenerator(terrain_size=(8.0, 8.0), resolution=0.05)
    
    # Test different difficulty levels
    for difficulty in [0, 3, 6, 9]:
        generator.set_difficulty(difficulty)
        print(f"\nGenerating terrain with difficulty {difficulty}...")
        
        # Generate mixed terrain
        height_map = generator.generate_mixed_terrain(seed=42)
        height_map = generator.smooth_terrain(height_map, iterations=2)
        
        print(f"  Height map shape: {height_map.shape}")
        print(f"  Min height: {height_map.min():.4f}m")
        print(f"  Max height: {height_map.max():.4f}m")
        print(f"  Mean height: {height_map.mean():.4f}m")
        print(f"  Std dev: {height_map.std():.4f}m")
        
        # Visualize (optional)
        # generator.visualize_terrain(height_map, f"terrain_difficulty_{difficulty}.png")
    
    print("\n" + "=" * 60)
    print("Demo complete! Terrain generation is working.")
