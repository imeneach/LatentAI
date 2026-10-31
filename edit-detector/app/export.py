import numpy as np


def _fmt(key: str, v) -> str:
    v = float(v)
    return f"{v:+.2f}" if key == "Exposure2012" else str(round(v))


def xmp_preset(params: dict, name: str = "Detected edit") -> str:
    attrs = "\n".join(f'   crs:{k}="{_fmt(k, v)}"' for k, v in params.items())
    return f"""<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
   xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:PresetType="Normal"
   crs:Cluster=""
   crs:UUID="00000000000000000000000000000000"
   crs:SupportsAmount="False"
   crs:SupportsColor="True"
   crs:SupportsMonochrome="True"
   crs:SupportsHighDynamicRange="True"
   crs:SupportsNormalDynamicRange="True"
   crs:SupportsSceneReferred="True"
   crs:SupportsOutputReferred="True"
   crs:Version="15.0"
   crs:ProcessVersion="11.0"
   crs:HasSettings="True"
{attrs}>
   <crs:Name><rdf:Alt><rdf:li xml:lang="x-default">{name}</rdf:li></rdf:Alt></crs:Name>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"""


def cube(luts: list[np.ndarray], size: int = 17, title: str = "Detected edit") -> str:
    """3D .cube LUT from per-channel curves (red varies fastest, as the format requires)."""
    idx = np.rint(np.arange(size) * 255 / (size - 1)).astype(int)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}"]
    for b in idx:
        for g in idx:
            for r in idx:
                lines.append(f"{luts[0][r] / 255:.5f} {luts[1][g] / 255:.5f} {luts[2][b] / 255:.5f}")
    return "\n".join(lines) + "\n"
