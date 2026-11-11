"""
PhaseUNet architecture diagram (three-row symmetric layout), matplotlib-based.
Produces both .pdf and .png in the paper directory.

Layout:
  Row 1 (top, L->R):     INPUT | UNET BACKBONE | (s,c) | PHASE HEAD
  Row 2 (middle):                                 phi  | k_gt
  Row 3 (bottom, R->L):  DEPTH | CALIBRATION  | Phihat | UNWRAP

Column widths are matched between row 1 and row 3:
  Col 1 (I/O):      Input <-> Depth
  Col 2 (math grp): UNet  <-> Calibration  (both 3 sub-blocks)
  Col 3 (tensor):   (s,c) <-> Phihat
  Col 4 (op grp):   Phase <-> Unwrap
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# STIX gives a Times/LaTeX-like look with heavier strokes than Computer
# Modern, so equations stay legible when the figure is shrunk to column
# width in the paper PDF.
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral', 'DejaVu Serif', 'Times New Roman', 'serif']


# ----------------------------------------------------------------------
# Layout constants (canvas units -- 1 unit ~ 1 inch at figsize=figaspect)
# ----------------------------------------------------------------------
CANVAS_W = 16.0
# Canvas height is set just above the top of the tallest row-1 group
# (UNet Backbone top = Y_ROW1 + H_GROUP3/2 = 6.75 + 1.5 = 8.25) with a small
# margin. Keeps whitespace above the UNet block to a minimum.
CANVAS_H = 8.6

# Matched column widths (strict: row 1 and row 3 share these)
W_IO         = 2.2   # Col 1: Input / Depth
W_GROUP_MATH = 3.0   # Col 2: UNet Backbone / Calibration
W_TENSOR     = 2.0   # Col 3: (s,c) / Phihat
W_GROUP_OP   = 2.6   # Col 4: Phase Head / Unwrap
W_AUX        = 2.4   # k_gt (row 2 only)

# Constant horizontal gap between adjacent columns
GAP_COL = 0.6

# Column left-edges (computed so gaps are constant)
X1     = 0.4
X2     = X1 + W_IO         + GAP_COL  # = 3.2
X3     = X2 + W_GROUP_MATH + GAP_COL  # = 6.8
X4     = X3 + W_TENSOR     + GAP_COL  # = 9.4
X_AUX  = X4 + W_GROUP_OP   + GAP_COL  # = 12.6

# Row center y-coordinates.
# Chosen so the two vertical arrows (Phase Head -> phi, phi -> Unwrap) have
# identical length L_ARR = 0.6, accounting for the different half-heights of
# Phase Head (H_GROUP2/2 = 1.15) vs. Unwrap (H_GROUP1/2 = 0.75) and the phi
# tensor (H_TENSOR/2 = 0.675).
#   Y_ROW1 - Y_ROW2 = L_ARR + H_GROUP2/2 + H_TENSOR/2 = 0.6 + 1.15 + 0.675 = 2.425
#   Y_ROW2 - Y_ROW3 = L_ARR + H_TENSOR/2 + H_GROUP1/2 = 0.6 + 0.675 + 0.75 = 2.025
Y_ROW3 = 2.3
Y_ROW2 = Y_ROW3 + 2.025   # = 4.325
Y_ROW1 = Y_ROW2 + 2.425   # = 6.75

# Heights per box type
H_IO        = 1.8
H_GROUP3    = 3.0   # groups with 3 sub-blocks + header (UNet, Calibration)
H_GROUP2    = 2.3   # groups with 2 sub-blocks + header (Phase Head)
H_GROUP1    = 1.5   # groups with 1 sub-block + header (Unwrap)
H_TENSOR    = 1.35
H_AUX       = 1.35

# Sub-block style
SUB_H       = 0.60
SUB_GAP     = 0.18  # vertical gap between sub-blocks (leaves room for arrow)
HDR_PAD     = 0.55  # vertical padding from top of group to header baseline

# Colors
COLORS = {
    'input':   '#BBDEFB',  # light blue
    'output':  '#B2DFDB',  # teal
    'unet':    '#FFE0B2',  # light orange
    'phase':   '#F8BBD0',  # light pink
    'unwrap':  '#F8BBD0',  # same pink family (phase operation)
    'cal':     '#C8E6C9',  # light green
    'tensor':  '#C8E6C9',  # same green family (intermediate tensors)
    'aux':     '#FFF59D',  # light yellow
    'border':  '#333333',
    'arrow':   '#555555',
    'text':    '#1a1a1a',
    # Group header text colors (match group fill tone)
    'unet_hdr':  '#8B5A00',
    'phase_hdr': '#AD1457',
    'unwrap_hdr':'#AD1457',
    'cal_hdr':   '#1B5E20',
}

# Font sizes — sized so math stays legible once the figure is scaled to
# column width in the paper.
FS_HDR   = 16   # group headers ("UNET BACKBONE", etc.)
FS_IO    = 16   # I/O box text
FS_TENSOR= 15
FS_SUB   = 14   # sub-block text
FS_AUX   = 15
FS_LOSS  = 24


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def draw_box(ax, x, y, w, h, text, color, fontsize=12, bold=False,
             text_color=None, boxstyle="round,pad=0.02,rounding_size=0.15"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=boxstyle,
        facecolor=color,
        edgecolor=COLORS['border'],
        linewidth=1.4
    )
    ax.add_patch(box)
    if text:
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight,
                color=text_color or COLORS['text'])


def draw_group_with_header(ax, x, y, w, h, header_text, fill_color,
                            header_color, sublabels, hdr_fs=FS_HDR,
                            sub_fs=FS_SUB):
    """
    Draw a coloured group box with a bold header near the top and
    `sublabels` as white rounded rectangles stacked below the header.
    Returns the list of (sub_x, sub_y, sub_w, sub_h) of sub-blocks
    so caller can draw inter-sub arrows.
    """
    # Group container
    draw_box(ax, x, y, w, h, '', fill_color,
             boxstyle="round,pad=0.02,rounding_size=0.18")
    # Header text
    hdr_y = y + h - 0.32
    ax.text(x + w/2, hdr_y, header_text,
            ha='center', va='center',
            fontsize=hdr_fs, fontweight='bold',
            color=header_color)

    # Sub-blocks: center them horizontally; stack from top to bottom
    n = len(sublabels)
    sub_w = w - 0.5
    sub_x = x + (w - sub_w) / 2
    stack_top_y = y + h - HDR_PAD - SUB_H  # y of FIRST (topmost) sub-block

    positions = []
    for i, label in enumerate(sublabels):
        sub_y = stack_top_y - i * (SUB_H + SUB_GAP)
        draw_box(ax, sub_x, sub_y, sub_w, SUB_H, label, '#FFFFFF',
                 fontsize=sub_fs,
                 boxstyle="round,pad=0.02,rounding_size=0.08")
        positions.append((sub_x, sub_y, sub_w, SUB_H))

    # Inter-sub arrows (small, vertical)
    for i in range(n - 1):
        x_center = positions[i][0] + positions[i][2] / 2
        y_from = positions[i][1]                       # bottom of sub i
        y_to   = positions[i+1][1] + positions[i+1][3] # top of sub i+1
        ax.add_patch(FancyArrowPatch(
            (x_center, y_from), (x_center, y_to),
            arrowstyle='-|>', mutation_scale=10,
            color=COLORS['arrow'], linewidth=0.9
        ))

    return positions


def draw_arrow(ax, x1, y1, x2, y2, linestyle='-', color=None, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=18,
        color=color or COLORS['arrow'],
        linewidth=lw, linestyle=linestyle
    ))


def draw_bent_arrow(ax, points, linestyle='-', color=None, lw=1.6):
    """Draw a polyline through `points` with an arrowhead at the end."""
    col = color or COLORS['arrow']
    for i in range(len(points) - 1):
        ax.plot([points[i][0], points[i+1][0]],
                [points[i][1], points[i+1][1]],
                color=col, linewidth=lw, linestyle=linestyle,
                solid_capstyle='round')
    # Final arrowhead segment
    px, py = points[-2]
    qx, qy = points[-1]
    dx, dy = qx - px, qy - py
    ax.add_patch(FancyArrowPatch(
        (px + dx * 0.7, py + dy * 0.7), (qx, qy),
        arrowstyle='-|>', mutation_scale=18,
        color=col, linewidth=lw, linestyle=linestyle
    ))


# ----------------------------------------------------------------------
# Main drawing function
# ----------------------------------------------------------------------
def draw_phaseunet_architecture(output_path_pdf, output_path_png=None):
    fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H), facecolor='white')
    ax.set_aspect('equal')
    ax.axis('off')

    # ==================== ROW 1 (top, L -> R) ====================
    # INPUT (col 1) -- raw fringe image
    y_input = Y_ROW1 - H_IO/2
    draw_box(ax, X1, y_input, W_IO, H_IO,
             r'$\mathbf{Input\ fringe}$'
             + '\n' + r'$I(u,v)$'
             + '\n' + r'$[B, 1, H, W]$',
             COLORS['input'], fontsize=FS_IO, bold=False)

    # UNET BACKBONE (col 2, 3 sub-blocks; learnable feature extractor)
    y_unet = Y_ROW1 - H_GROUP3/2
    draw_group_with_header(
        ax, X2, y_unet, W_GROUP_MATH, H_GROUP3,
        'UNET BACKBONE',
        COLORS['unet'], COLORS['unet_hdr'],
        [r'Encoder: $I \rightarrow$ feats',
         r'Bottleneck',
         r'Decoder + Conv $1{\times}1$']
    )

    # (s, c) tensor (col 3) -- raw output of UNet before normalization
    y_sc = Y_ROW1 - H_TENSOR/2
    draw_box(ax, X3, y_sc, W_TENSOR, H_TENSOR,
             r'$(s, c) = f_\theta(I)$'
             + '\n' + r'$[B, 2, H, W]$',
             COLORS['tensor'], fontsize=FS_TENSOR, bold=False)

    # PHASE HEAD (col 4, 2 sub-blocks; L2 projection then atan2)
    y_phase = Y_ROW1 - H_GROUP2/2
    draw_group_with_header(
        ax, X4, y_phase, W_GROUP_OP, H_GROUP2,
        'PHASE HEAD',
        COLORS['phase'], COLORS['phase_hdr'],
        [r'$(\hat{s},\hat{c}) = \dfrac{(s,c)}{\sqrt{s^2+c^2+\varepsilon}}$',
         r'$\hat{\phi} = \mathrm{atan2}(\hat{s},\hat{c})$']
    )

    # Row 1 horizontal arrows (L -> R, at y = Y_ROW1)
    draw_arrow(ax, X1 + W_IO,         Y_ROW1, X2,                        Y_ROW1)
    draw_arrow(ax, X2 + W_GROUP_MATH, Y_ROW1, X3,                        Y_ROW1)
    draw_arrow(ax, X3 + W_TENSOR,     Y_ROW1, X4,                        Y_ROW1)

    # ==================== ROW 2 (middle) ====================
    # phi tensor (aligned under Phase Head column) -- wrapped phase
    x_phi = X4 + (W_GROUP_OP - W_TENSOR) / 2
    y_phi = Y_ROW2 - H_TENSOR/2
    draw_box(ax, x_phi, y_phi, W_TENSOR, H_TENSOR,
             r'$\hat{\phi}(u,v)$'
             + '\n' + r'$\in [-\pi, \pi)$',
             COLORS['tensor'], fontsize=FS_TENSOR, bold=False)

    # k_gt auxiliary -- oracle fringe order (integer)
    y_kgt = Y_ROW2 - H_AUX/2
    draw_box(ax, X_AUX, y_kgt, W_AUX, H_AUX,
             r'$\mathbf{Oracle\ fringe\ order}$'
             + '\n' + r'$k_{\mathrm{gt}}(u,v)$'
             + '\n' + r'$\in \{0, \ldots, K{-}1\}$',
             COLORS['aux'], fontsize=FS_AUX, bold=False)

    # Phase Head (row 1) -> phi (row 2): vertical
    x_col4_center = X4 + W_GROUP_OP/2
    draw_arrow(ax, x_col4_center, y_phase, x_col4_center, y_phi + H_TENSOR)

    # ==================== ROW 3 (bottom, R -> L) ====================
    # OUTPUT: Predicted depth (col 1)
    y_depth = Y_ROW3 - H_IO/2
    draw_box(ax, X1, y_depth, W_IO, H_IO,
             r'$\mathbf{Predicted\ depth}$'
             + '\n' + r'$\hat{D}(u,v)$'
             + '\n' + r'$[B, 1, H, W]$',
             COLORS['output'], fontsize=FS_IO, bold=False)

    # CALIBRATION (col 2, 3 sub-blocks with explicit calibration math)
    #   1) phase --> projector column
    #   2) stereo triangulation (intersect camera ray and projector plane)
    #   3) take z-coordinate as depth
    y_cal = Y_ROW3 - H_GROUP3/2
    draw_group_with_header(
        ax, X2, y_cal, W_GROUP_MATH, H_GROUP3,
        'CALIBRATION',
        COLORS['cal'], COLORS['cal_hdr'],
        [r'$v_p = \dfrac{T_p}{2\pi}\,\hat{\Phi}$',
         r'$(X,Y,Z) = \mathrm{triangulate}(u,v,v_p)$',
         r'$\hat{D} = Z$']
    )

    # Phihat tensor (col 3) -- unwrapped phase, defines the equation
    y_phihat = Y_ROW3 - H_TENSOR/2
    draw_box(ax, X3, y_phihat, W_TENSOR, H_TENSOR,
             r'$\hat{\Phi}(u,v)$'
             + '\n' + r'unwrapped',
             COLORS['tensor'], fontsize=FS_TENSOR, bold=False)

    # UNWRAP (col 4, 1 sub-block showing the unwrap equation)
    y_unwrap = Y_ROW3 - H_GROUP1/2
    draw_group_with_header(
        ax, X4, y_unwrap, W_GROUP_OP, H_GROUP1,
        'UNWRAP',
        COLORS['unwrap'], COLORS['unwrap_hdr'],
        [r'$\hat{\Phi} = \hat{\phi} + 2\pi\,k_{\mathrm{gt}}$']
    )

    # Row 3 horizontal arrows (R -> L at y = Y_ROW3)
    # Unwrap -> Phihat
    draw_arrow(ax, X4,                        Y_ROW3, X3 + W_TENSOR,     Y_ROW3)
    # Phihat -> Calibration
    draw_arrow(ax, X3,                        Y_ROW3, X2 + W_GROUP_MATH, Y_ROW3)
    # Calibration -> Depth
    draw_arrow(ax, X2,                        Y_ROW3, X1 + W_IO,         Y_ROW3)

    # phi (row 2) -> Unwrap (row 3): vertical
    draw_arrow(ax, x_col4_center, y_phi, x_col4_center, y_unwrap + H_GROUP1)

    # k_gt -> Unwrap (L-shape, dashed, auxiliary)
    x_kgt_center = X_AUX + W_AUX/2
    bent = [
        (x_kgt_center, y_kgt),          # bottom of k_gt
        (x_kgt_center, Y_ROW3),         # down to row 3 center y
        (X4 + W_GROUP_OP, Y_ROW3),      # left to right edge of UNWRAP
    ]
    draw_bent_arrow(ax, bent, linestyle='--', color='#8B7500', lw=1.3)

    # ==================== Training loss (below row 3) ====================
    # Center over the full visible content (left edge of Input ... right edge
    # of k_gt auxiliary) so the equation reads as belonging to the whole fig.
    loss_y = Y_ROW3 - H_GROUP3/2 - 0.45
    loss_x = (X1 + X_AUX + W_AUX) / 2
    ax.text(loss_x, loss_y,
            r'$\mathbf{Training\ loss:}\ \ \mathcal{L} = '
            r'w_c\,\mathcal{L}_{\mathrm{circ}} + w_g\,\mathcal{L}_{\nabla} + '
            r'w_d\,\mathcal{L}_d\ \ '
            r'\mathrm{with}\ \ (w_c, w_g, w_d) = (1.0,\, 0.5,\, 0.1)$',
            ha='center', va='center',
            fontsize=FS_LOSS, color='#555555')

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight',
                facecolor='white')
    print(f"Saved: {output_path_pdf}")
    if output_path_png:
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f"Saved: {output_path_png}")
    plt.close()


if __name__ == "__main__":
    paper_dir = Path(__file__).resolve().parent.parent / 'paper'
    paper_dir.mkdir(exist_ok=True)
    pdf_path = paper_dir / 'phaseunet_architecture.pdf'
    png_path = paper_dir / 'phaseunet_architecture.png'
    draw_phaseunet_architecture(str(pdf_path), str(png_path))
