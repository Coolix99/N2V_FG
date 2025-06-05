import torch



def test_unet_output_shape():
    from n2v_fg.unet import UNet2D
    model = UNet2D(in_channels=60, out_channels=60, base_channels=32, depth=2)
    dummy_input = torch.randn(1, 60, 64, 64)  # B=1, C=60 (e.g. 5 z × 3 color × 4 time)
    output = model(dummy_input)
    assert output.shape == (1, 60, 64, 64), f"Unexpected shape: {output.shape}"

def test_unet_output_shape_GN():
    from n2v_fg.unet import UNet2D_GN as UNet2D
    model = UNet2D(in_channels=60, out_channels=60, base_channels=32, depth=2)
    dummy_input = torch.randn(1, 60, 64, 64)  # B=1, C=60 (e.g. 5 z × 3 color × 4 time)
    output = model(dummy_input)
    assert output.shape == (1, 60, 64, 64), f"Unexpected shape: {output.shape}"

if __name__ == "__main__":
    test_unet_output_shape()
    test_unet_output_shape_GN()