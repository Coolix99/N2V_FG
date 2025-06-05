import torch



def test_unet_output_shape():
    from n2v_fg.net import SpatioTemporalDenoiser
    model = SpatioTemporalDenoiser(num_z=4,k_t=5)
    dummy_input = torch.randn(1,100, 4, 64, 64)  #
    output = model(dummy_input)
    assert output.shape == (1,100, 4, 64, 64), f"Unexpected shape: {output.shape}"


if __name__ == "__main__":
    test_unet_output_shape()
