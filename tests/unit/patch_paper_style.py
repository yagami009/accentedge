"""Patch script for paper_style.py"""

from pathlib import Path

p = Path('/Users/ayushmh/accentedge-model-lab/src/accentedge_lab/models/streaming_ac/paper_style.py')
text = p.read_text()

old = '''    def forward(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.norm(x)

        residual = x
        for layer in self.layers:'''

new = '''    def forward(
        self, x: torch.Tensor, state: Optional[dict] = None
    ) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)

        B, T, _ = x.shape
        x = x.reshape(B * T, -1)
        x = self.input_proj(x)
        x = self.norm(x)
        x = x.reshape(B, T, -1)

        residual = x
        for layer in self.layers:'''

if old in text:
    text = text.replace(old, new)
    p.write_text(text)
    print('patched successfully')
else:
    print('old pattern not found')
