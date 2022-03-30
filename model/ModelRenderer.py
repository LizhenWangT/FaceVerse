import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    TexturesVertex,
    blending
)

class ModelRenderer:
    def __init__(self, focal=1315, img_size=224, device='cuda:0'):
        self.img_size = img_size
        self.focal = focal
        self.device = device

        self.alb_renderer = self._get_renderer(albedo=True)
        self.sha_renderer = self._get_renderer(albedo=False)

    def _get_renderer(self, albedo=True):
        R, T = look_at_view_transform(10, 0, 0)  # camera's position
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01, zfar=50,
                                        fov=2 * np.arctan(self.img_size // 2 / self.focal) * 180. / np.pi)

        if albedo:
            lights = PointLights(device=self.device, location=[[0.0, 0.0, 1e5]],
                                ambient_color=[[1, 1, 1]],
                                specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])
        else:
            lights = PointLights(device=self.device, location=[[0.0, 0.0, 1e5]],
                                ambient_color=[[0.1, 0.1, 0.1]],
                                specular_color=[[0.0, 0.0, 0.0]], diffuse_color=[[0.95, 0.95, 0.95]])

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardFlatShader(
                device=self.device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer

