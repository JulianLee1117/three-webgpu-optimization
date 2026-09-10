# Glass Vault artwork

`assets/material-atlas.png` is an original generated runtime base-color atlas, used directly on the 3D vault, its engraved trim, and velvet stage. It is not a concept-image backdrop, a photograph, or measured physically based material data. No generated pixels are used for the optical phase or receiver intensity.

- Tool: built-in `image_gen.imagegen`; the tool does not report an exact model identifier.
- Generated source: `exec-724cafb5-503b-4d46-9c36-5dd42436ff5b.png`.
- Saved output: `assets/material-atlas.png`, unchanged from the generated PNG.
- Dimensions: 1254 × 1254 pixels; 3,782,141 bytes.
- SHA-256: `b0d0a7cca9a4568ea5ed1437d9044c88fed7fac3931566a89f0935c24f548d66`.
- Quadrants: top left obsidian; top right engraved gold; bottom left velvet; bottom right gold rosette on obsidian.
- The runtime selects quadrants by texture UV transforms with a half-texel inset. Decorative engraving is artwork; metallic response and roughness are separately authored scalar material settings.

## Exact generation prompt

```text
Use case: stylized-concept
Asset type: A square production texture atlas for a beautiful interactive 3D obsidian and brass optical vault. This image will be sampled directly onto mesh surfaces, so it must be flat material artwork, NOT a picture of a room or a rendered object.
Primary request: Perfect 2 by 2 grid of four equal square material tiles, with no margins, no border, no gutters, no lettering. Quadrants meet exactly at image midlines.
Top-left quadrant: Deep black obsidian with extremely subtle midnight-teal mineral veins, a nearly black polished-stone base-color texture. Fine restrained mineral detail, no strong reflected white lights, no rendered perspective, no apparent sphere or block.
Top-right quadrant: Antique satin gold with intricate fine botanical and art-deco engraved filigree. Dense delicate leaf scrolls, geometric arcs, and minute etched linework. A flat decorative brass surface, rich but restrained, with even diffuse illumination and no shiny white hotspot.
Bottom-left quadrant: Midnight deep-teal velvet fabric, very dark and luxurious, fine understated soft woven texture, evenly lit, no draped cloth folds, no perspective.
Bottom-right quadrant: An exquisite circular radial rosette of fine gold filigree engraved onto black obsidian. One centered complete circular motif, elegant botanical scrolls and celestial geometric arcs, no words and no numbers. Thin gold detail, large open dark center, motif contained within its square, flat orthographic decorative design.
Style: Museum-quality craftsmanship, rich tactile materials, elegant Art Deco scientific instrument ornamentation. Very fine crisp details that remain clean when applied as 3D textures. Top-left and bottom-left should stay dark. No baked spotlights, no cast shadows from unrelated objects.
Constraints: Exactly equal quadrant sizes. No text, no watermark, no frame, no margins, no mockup, no object render, no scenery, no perspective.
```

## Display geometry

The object is an interactive schematic. Its lateral dimensions and visible spacing are magnified for readability and do not represent millimetre-scale fabrication geometry. The analytical propagation is owned by the experiment's solver. The source plate remains fixed at world `[-2.4, 1.65, 0]`, normal `+X`, even when the vault door opens. The receiver is centered at `[receiverX, 1.65, 0]` and faces `-X`. Materials assigned by the main application to `phaseMesh` or `receiverMesh` remain the main application's disposal responsibility.
