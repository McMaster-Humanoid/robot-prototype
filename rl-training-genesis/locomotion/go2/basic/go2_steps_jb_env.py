import numpy as np
import genesis as gs


class Steps_Environment:
    def __init__(
        self,
        scene,
        seed=42,
        map_size=40,
        platform_size=8,
        gap=1,
        max_height=20,
        horizontal_scale=0.05,
        vertical_scale=0.005,
    ):
        self.scene = scene
        self.seed = seed
        self.map_size = map_size
        self.platform_size = platform_size
        self.gap = gap
        self.max_height = max_height
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale

        self.terrain = self._add_terrain()

    def _build_heightfield(self):
        hf = np.zeros((self.map_size, self.map_size), dtype=np.int16)
        rng = np.random.default_rng(self.seed)

        step = self.platform_size + self.gap
        for row in range(0, hf.shape[0], step):
            for col in range(0, hf.shape[1], step):
                r_end = min(row + self.platform_size, hf.shape[0])
                c_end = min(col + self.platform_size, hf.shape[1])
                hf[row:r_end, col:c_end] = int(rng.integers(0, self.max_height + 1))

        return hf

    def _add_terrain(self):
        total_size = self.map_size * self.horizontal_scale
        return self.scene.add_entity(
            morph=gs.morphs.Terrain(
                height_field=self._build_heightfield(),
                horizontal_scale=self.horizontal_scale,
                vertical_scale=self.vertical_scale,
                pos=(-total_size / 2, -total_size / 2, 0.0),
            )
        )


if __name__ == "__main__":
    gs.init(seed=0, backend=gs.gpu)

    scene = gs.Scene(show_viewer=True)
    Steps_Environment(
        scene=scene,
        seed=42,
        map_size=40,
        platform_size=8,
        gap=1,
        max_height=20,
        horizontal_scale=0.2,
        vertical_scale=0.005,
    )
    scene.build()
    for _ in range(2_000):
        scene.step()
