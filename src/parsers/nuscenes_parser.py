class NuScenesParser:
    def __init__(self, version, dataroot):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.scenes = self._load_scenes()
        self.samples = self._load_samples()
        self.annotations = self._load_annotations()