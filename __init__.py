"""MMH3Tools - MiniMax H3 latent tooling for chained long-form AV generation."""

from comfy_api.latest import ComfyExtension

from .mmh3tools import NODES


class MMH3ToolsExtension(ComfyExtension):
    async def get_node_list(self):
        return NODES


async def comfy_entrypoint() -> MMH3ToolsExtension:
    return MMH3ToolsExtension()


# serves web/js/:
#   mmh3_dimension_calculator.js -- repopulates the resolution dropdown from
#     /mmh3-dim-calc/resolutions when ratio or orientation changes
#   mmh3_live_preview.js -- the DOM widget MMH3 Live Preview draws its filmstrip
#     on. Needed because core's preview channel always addresses the EXECUTING
#     node, so previews cannot otherwise be told apart.
WEB_DIRECTORY = "./web/js"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
