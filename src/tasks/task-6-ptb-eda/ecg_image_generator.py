import numpy as np
import plotly.graph_objects as go


class ECGImageGenerator(object):
    def __init__(
            self,
            millimeters_per_second=25,
            vertical_separation_between_leads=20, # millimeters
            vertical_margin=10, # millimeters
            millimeters_per_millivolt=10,
            leads_order=None,
    ):
        self.millimeters_per_second = millimeters_per_second
        self.vertical_separation_between_leads = vertical_separation_between_leads
        self.vertical_margin = vertical_margin
        self.millimeters_per_millivolt = millimeters_per_millivolt

        if leads_order is None:
            leads_order = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        self.leads_order = leads_order

    def _ecg_to_millimeters(self, ecg, ecg_metadata):
        vertical_offsets, height = self._get_vertical_offsets(ecg_metadata)
        lead_names = ecg_metadata["sig_name"]

        ecg_millimeters = {}
        for lead_name, lead_millivolts in zip(lead_names, ecg.T):
            if lead_name not in self.leads_order:
                continue

            lead_millimiters = vertical_offsets[lead_name] + lead_millivolts * self.millimeters_per_millivolt
            ecg_millimeters[lead_name] = lead_millimiters

        return ecg_millimeters, height

    def _get_vertical_offsets(self, ecg_metadata):
        num_leads = len(self.leads_order)
        lead_names = ecg_metadata["sig_name"]

        height = self.vertical_separation_between_leads * (num_leads - 1) + 2 * self.vertical_margin

        vertical_offsets = {}
        for lead_name in lead_names:
            if lead_name not in self.leads_order:
                continue

            lead_order = self.leads_order.index(lead_name)
            vertical_offset = height - self.vertical_margin - lead_order * self.vertical_separation_between_leads

            vertical_offsets[lead_name] = vertical_offset

        return vertical_offsets, height

    def _get_time_vector(self, ecg_metadata):
        num_samples = ecg_metadata["sig_len"]
        sampling_frequency = ecg_metadata["fs"]
        millimeters_per_sample = self.millimeters_per_second / sampling_frequency
        time_vector = np.arange(num_samples) * millimeters_per_sample

        duration = num_samples / sampling_frequency
        width = duration * self.millimeters_per_second

        return time_vector, width

    @staticmethod
    def _get_clean_layout(height_millimeters, width_millimeters):
        width_pixels = width_millimeters * 3
        height_pixels = height_millimeters * 3

        return go.Layout(
            xaxis=go.layout.XAxis(
                range=[0, width_millimeters],
                constrain="domain",
                showticklabels=False,
            ),
            yaxis=go.layout.YAxis(
                range=[0, height_millimeters],
                tickmode='linear',
                scaleanchor="x",
                scaleratio=1,
                showticklabels=False,
                constrain="domain",
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=0  # top margin
            ),
            width=width_pixels,
            height=height_pixels,
        )

    def plot_ecg(self, ecg, ecg_metadata, output=None, clean_generation=False):
        ecg_millimeters, height_millimeters = self._ecg_to_millimeters(ecg, ecg_metadata)

        time_vector, width_millimeters = self._get_time_vector(ecg_metadata)

        layout = self._get_clean_layout(height_millimeters, width_millimeters)
        if not clean_generation:
            layout.update(
                xaxis=go.layout.XAxis(
                    range=[0, width_millimeters],
                    constrain="domain",
                    showticklabels=False,
                    gridcolor='LightPink',
                    tickmode='linear',
                    tick0=0,
                    dtick=5,
                ),
                yaxis=go.layout.YAxis(
                    range=[0, height_millimeters],
                    gridcolor='LightPink',
                    tickmode='linear',
                    tick0=0,
                    dtick=5,
                    scaleanchor="x",
                    scaleratio=1,
                    showticklabels=False,
                    constrain="domain",
                ),
            )

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=time_vector,
                    y=lead,
                    name=name,
                    marker=dict(color="black"),
                    showlegend=False
                )
                for name, lead in ecg_millimeters.items()
            ],
            layout=layout
        )

        vertical_offsets, _ = self._get_vertical_offsets(ecg_metadata)

        for lead, offset in vertical_offsets.items():
            fig.add_annotation(
                x=5,
                y=offset + 5,
                text=lead,
                showarrow=False,
                xanchor="left"
            )

        if output:
            fig.write_image(output, width=layout.width, height=layout.height, scale=1)

        return fig
