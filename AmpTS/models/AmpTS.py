from typing import Tuple

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("moving-average kernel must be a positive odd integer")
        self.pool = nn.AvgPool1d(self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        if pad > 0:
            front = x[:, :1, :].expand(-1, pad, -1)
            end = x[:, -1:, :].expand(-1, pad, -1)
            x = torch.cat((front, x, end), dim=1)
        return self.pool(x.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        fluctuation = x - trend
        return trend, fluctuation


class TemporalConvolution(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("conv_kernel must be a positive odd integer")
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        if input_channels == output_channels:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
            center = kernel_size // 2
            with torch.no_grad():
                for channel in range(input_channels):
                    self.conv.weight[channel, channel, center] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class DecompositionProjection(nn.Module):
    def __init__(self, input_length: int, output_length: int, moving_avg: int) -> None:
        super().__init__()
        self.decomposition = SeriesDecomposition(moving_avg)
        self.trend_projection = nn.Linear(input_length, output_length)
        self.fluctuation_projection = nn.Linear(input_length, output_length)
        nn.init.constant_(self.trend_projection.weight, 1.0 / input_length)
        nn.init.zeros_(self.trend_projection.bias)
        nn.init.constant_(self.fluctuation_projection.weight, 1.0 / input_length)
        nn.init.zeros_(self.fluctuation_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend, fluctuation = self.decomposition(x)
        trend = self.trend_projection(trend.transpose(1, 2)).transpose(1, 2)
        fluctuation = self.fluctuation_projection(fluctuation.transpose(1, 2)).transpose(1, 2)
        return trend + fluctuation


class ConvLinearHistoryEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_channels: int,
        output_channels: int,
        conv_kernel: int,
    ) -> None:
        super().__init__()
        self.temporal_conv = TemporalConvolution(input_channels, output_channels, conv_kernel)
        self.temporal_projection = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        return self.temporal_projection(x.transpose(1, 2)).transpose(1, 2)


class ConvDecompHistoryEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_channels: int,
        output_channels: int,
        conv_kernel: int,
        moving_avg: int,
    ) -> None:
        super().__init__()
        if output_channels > input_channels:
            raise ValueError("c_out must not exceed enc_in")
        self.output_channels = int(output_channels)
        self.temporal_conv = TemporalConvolution(input_channels, input_channels, conv_kernel)
        self.decomposition_projection = DecompositionProjection(seq_len, pred_len, moving_avg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.decomposition_projection(x)
        if x.shape[-1] != self.output_channels:
            x = x[:, :, -self.output_channels:]
        return x


class ConvLinearDecompHistoryEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_channels: int,
        output_channels: int,
        conv_kernel: int,
        moving_avg: int,
    ) -> None:
        super().__init__()
        self.conv_linear = ConvLinearHistoryEncoder(
            seq_len,
            pred_len,
            input_channels,
            output_channels,
            conv_kernel,
        )
        self.decomposition_projection = DecompositionProjection(pred_len, pred_len, moving_avg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decomposition_projection(self.conv_linear(x))


class StepConditionedRestorationNetwork(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        input_channels: int,
        output_channels: int,
        alpha_bar: torch.Tensor,
        history_mode: str,
        conv_kernel: int,
        moving_avg: int,
    ) -> None:
        super().__init__()
        self.eps = 1e-4
        self.gamma = nn.Parameter(alpha_bar.float().clone())
        history_mode = str(history_mode).lower()

        if history_mode == "conv_linear":
            self.history_encoder = ConvLinearHistoryEncoder(
                seq_len, pred_len, input_channels, output_channels, conv_kernel
            )
        elif history_mode == "conv_decomp":
            self.history_encoder = ConvDecompHistoryEncoder(
                seq_len,
                pred_len,
                input_channels,
                output_channels,
                conv_kernel,
                moving_avg,
            )
        elif history_mode == "conv_linear_decomp":
            self.history_encoder = ConvLinearDecompHistoryEncoder(
                seq_len,
                pred_len,
                input_channels,
                output_channels,
                conv_kernel,
                moving_avg,
            )
        else:
            raise ValueError(
                "history_mode must be one of {'conv_linear', 'conv_decomp', 'conv_linear_decomp'}"
            )

        self.state_projection = nn.Linear(pred_len, pred_len)
        nn.init.eye_(self.state_projection.weight)
        nn.init.zeros_(self.state_projection.bias)

    def encode_history(self, normalized_history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(normalized_history)

    def forward(
        self,
        current_state: torch.Tensor,
        timesteps: torch.Tensor,
        historical_representation: torch.Tensor,
    ) -> torch.Tensor:
        gamma_t = self.gamma.gather(0, timesteps).view(-1, 1, 1)
        state_representation = self.state_projection(current_state.transpose(1, 2)).transpose(1, 2)
        fused_representation = historical_representation + state_representation
        denominator = torch.sqrt((1.0 - gamma_t).clamp_min(self.eps))
        return (
            gamma_t * current_state
            + (1.0 - 2.0 * gamma_t) * fused_representation
        ) / denominator


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        if configs.task_name != "long_term_forecast":
            raise NotImplementedError("AmpTS supports long-term forecasting in this release")

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.input_channels = int(configs.enc_in)
        self.output_channels = int(configs.c_out)
        self.restoration_steps = int(getattr(configs, "restoration_steps", 8))
        self.history_mode = str(getattr(configs, "history_mode", "conv_linear_decomp")).lower()
        self.conv_kernel = int(getattr(configs, "conv_kernel", 3))
        self.moving_avg = 25

        if self.output_channels > self.input_channels:
            raise ValueError("AmpTS requires c_out <= enc_in")
        if self.restoration_steps < 2:
            raise ValueError("restoration_steps must be at least 2")
        if self.conv_kernel <= 0 or self.conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be a positive odd integer")

        alpha_bar = 1.0 - torch.arange(
            self.restoration_steps + 1, dtype=torch.float32
        ) / float(self.restoration_steps)
        sqrt_alpha_bar = torch.sqrt(alpha_bar.clamp_min(0.0))
        sqrt_beta_bar = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))

        self.register_buffer("alpha_bar", alpha_bar, persistent=True)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar, persistent=True)
        self.register_buffer("sqrt_beta_bar", sqrt_beta_bar, persistent=True)

        self.scr_net = StepConditionedRestorationNetwork(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            alpha_bar=alpha_bar,
            history_mode=self.history_mode,
            conv_kernel=self.conv_kernel,
            moving_avg=self.moving_avg,
        )

    @staticmethod
    def _history_statistics(x_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - mean
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        return mean, stdev

    def _output_statistics(
        self, mean: torch.Tensor, stdev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.output_channels == self.input_channels:
            return mean, stdev
        return mean[:, :, -self.output_channels:], stdev[:, :, -self.output_channels:]

    def _normalize_history(
        self, x_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_enc = x_enc.float()
        mean, stdev = self._history_statistics(x_enc)
        return (x_enc - mean) / stdev, mean, stdev

    def _progressive_restoration(
        self,
        normalized_history: torch.Tensor,
        historical_representation: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = normalized_history.shape[0]
        current_state = normalized_history.new_zeros(
            batch_size, self.pred_len, self.output_channels
        )

        for step in range(self.restoration_steps, 0, -1):
            timesteps = torch.full(
                (batch_size,), step, device=current_state.device, dtype=torch.long
            )
            complete_future = self.scr_net(
                current_state,
                timesteps,
                historical_representation,
            )
            previous_step = step - 1
            a_t = self.sqrt_alpha_bar[step].view(1, 1, 1)
            b_t = self.sqrt_beta_bar[step].view(1, 1, 1).clamp_min(1e-8)
            residual_coordinate = (current_state - a_t * complete_future) / b_t
            a_previous = self.sqrt_alpha_bar[previous_step].view(1, 1, 1)
            b_previous = self.sqrt_beta_bar[previous_step].view(1, 1, 1)
            current_state = a_previous * complete_future + b_previous * residual_coordinate

        return current_state

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor = None,
        x_dec: torch.Tensor = None,
        x_mark_dec: torch.Tensor = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        normalized_history, mean, stdev = self._normalize_history(x_enc)
        output_mean, output_stdev = self._output_statistics(mean, stdev)
        historical_representation = self.scr_net.encode_history(normalized_history)
        prediction = self._progressive_restoration(
            normalized_history,
            historical_representation,
        )
        return prediction * output_stdev + output_mean
