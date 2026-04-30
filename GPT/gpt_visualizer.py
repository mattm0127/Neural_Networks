import torch
import pygame
from shakes_gpt import ShakesGPT, DataManager
from pathlib import Path

class ModelManager:

    def __init__(self, weight_path: str, data_path: str, prompt: str):

        self.weight_path = Path(weight_path)
        self.data_path = Path(data_path)
        self.prompt = prompt

        self.dm = DataManager(self.data_path)
        
        self.model = ShakesGPT(
            self.dm.tokenizer.vocab_size,
            embed_size=128,
            block_size=500,
            num_heads=4,
            num_layers=4
        ).to(self.dm.device)

        self.model.load_state_dict(
            torch.load(self.weight_path, map_location=self.dm.device, weights_only=True)
        )
        self.model.eval()

        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):

        def get_activation(name):

            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        self.model.token_embedding_table.register_forward_hook(get_activation('Token Embedding'))
        for i, block in enumerate(self.model.blocks):
            block.register_forward_hook(get_activation(f"Transformer Block {i+1}"))
        self.model.ln_f.register_forward_hook(get_activation('LN Final'))
        self.model.lm_head.register_forward_hook(get_activation("Head Output"))

    def run_inference(self)->dict[str, torch.Tensor]:
        e_prompt = self.dm.tokenizer.encode(self.prompt)
        t_prompt = self.dm.tokenizer.to_tensor(e_prompt)
        with torch.no_grad():
            logits, _ = self.model(t_prompt)
        
        return self.activations
    

class GPTVisualizer:

    WEIGHT_PATH = r"ShakesGPT.pth"
    DATA_PATH = r"LSTM\training_data\tiny_shakespear.txt"

    def __init__(self, prompt: str):
        pygame.init()
        self.prompt = prompt
        self.mm  = ModelManager(self.WEIGHT_PATH, self.DATA_PATH, self.prompt)
        self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        self.screen_rect = self.screen.get_rect()
        self.clock = pygame.time.Clock()
        self.activations = self.mm.run_inference()
        self.network_layers = {layer: self.create_matrix(layer) for layer in self.activations.keys()}
    
    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def create_matrix(self, activation):
        embed_size = self.activations[activation].size()[-1]
        block_size = len(self.prompt)

        act = self.activations[activation][0]
        act_min, act_max = act.min().item(), act.max().item()
        act_range = max(act_max - act_min, 1e-5)

        surface = pygame.Surface((3840, 2160))
        surface.fill((0, 0, 0))
        r_height = 2160 / block_size
        r_width = 3840 / embed_size

        for x in range(block_size):
            for y in range(embed_size):
                center_x = int(y * r_width + (r_width / 2))
                center_y = int(x * r_height + (r_height / 2))
                val = act[x, y].item()
                color_intensity = int(255 * (val - act_min) / act_range)
                color = (0 + color_intensity, 0 + color_intensity, 0 + color_intensity)
                if color_intensity > 125:
                    pygame.draw.circle(surface, color, (center_x, center_y), 12)
                else:
                    pygame.draw.circle(surface, color, (center_x, center_y), 12, 1)
        return surface

    def draw_title(self, text):
        font = pygame.font.SysFont('sysfont', 25)
        title = font.render(text, True, (125, 15, 220))
        title_rect = title.get_rect()
        title_rect.center = self.screen_rect.center
        title_rect.top = self.screen_rect.top + 10
        self.screen.blit(title, title_rect)

    def run_visualization(self):
        act_list = list(self.activations.keys())
        idx = 0
        timer = 1
        while True:
            self._check_events()
            self.screen.fill((0,0,0))
            screen_rect = self.screen.get_rect()
            layer = self.network_layers[act_list[idx]]
            t_layer = pygame.transform.smoothscale(layer, (self.screen.get_width()-20, self.screen.get_height()-80))
            t_layer_rect = t_layer.get_rect(center=screen_rect.center)
            self.screen.blit(t_layer, t_layer_rect)
            self.draw_title(act_list[idx])
            if timer % 60 == 0:
                idx = 0 if idx == len(act_list)-1 else idx + 1
                timer = 0
            timer += 1
            pygame.display.flip()
            self.clock.tick(60)


if __name__ == "__main__":
    gv = GPTVisualizer('MIAPUSS:\nThou sha')
    gv.run_visualization()
    