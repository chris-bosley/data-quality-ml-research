import numpy as np
import tools as datatools
import matplotlib.pyplot as plt


def make_metric_plots(metrics_for_plot, run_path, normalize=True):
    output_dataframe = datatools.pick_up_run_data(run_path)
    column_remap = {
        "mean_precision": "precision",
        "mean_recall": "recall",
        "mean_accuracy": "accuracy",
        "mean_f1": "f1",
        "mean_balanced-accuracy": "balanced-accuracy",
        "mean_roc_auc_score": "roc_auc_score",
        "mean_positive-retrieval": "positive-retrieval",
        "mean_negative-retrieval": "negative-retrieval",
        "mean_cost-penalty": "cost-penalty",
    }
    model_metrics = {}
    for model in np.unique(output_dataframe["model"].values):
        model_metrics[model] = {}
        for mutation in np.unique(
            output_dataframe[output_dataframe["model"] == model]["mutation"].values
        ):
            metrics_frame = (
                output_dataframe[
                    (output_dataframe["model"] == model)
                    & (output_dataframe["mutation"] == mutation)
                ]
                .set_index("severity")[column_remap.keys()]
                .rename(columns=column_remap)
            )
            model_metrics[model][mutation] = metrics_frame
    for model_to_plot in np.unique(output_dataframe["model"].values):
        for mutation in np.unique(output_dataframe["mutation"].values):
            # metrics_for_plot = ['precision', 'recall', 'f1', 'accuracy']
            if normalize:
                model_file = "_".join(model_to_plot.split() + [mutation, "normalized"])
                data_to_plot = (
                    model_metrics[model_to_plot][mutation]
                    / model_metrics[model_to_plot][mutation].iloc[0]
                )[metrics_for_plot]
                ymax = min(np.ceil(max(data_to_plot.max().values) * 10) / 10.0, 2.0)
            else:
                model_file = "_".join(model_to_plot.split() + [mutation])
                data_to_plot = model_metrics[model_to_plot][mutation][metrics_for_plot]
                ymax = 1
            plt.figure(figsize=(6, 6))
            data_to_plot.plot(figsize=[6, 6])
            plt.ylim([0, ymax])
            plt.xlim([0, 1])
            plt.xticks(np.arange(0, 1.01, 0.1))
            plt.yticks(np.arange(0, ymax + 0.01, 0.1))
            plt.grid(visible=True)
            if normalize:
                this_mutation = " ".join(
                    [t.title() for t in mutation.split("_")] + ["Normalized Metrics"]
                )
                plt.title(f"Model Metrics: {model_to_plot}: {this_mutation}")
            else:
                this_mutation = " ".join([t.title() for t in mutation.split("_")])
                plt.title(f"Model Metrics: {model_to_plot}: {this_mutation}")
            plt.ylabel("Metric Value")
            plt.xlabel("Mutation Severity")
            plt.legend(
                bbox_to_anchor=(0.5, -0.2),
                loc="lower center",
                ncol=len(metrics_for_plot),
            )
            # plt.tight_layout()
            plt.savefig(
                f"{run_path}/{model_file}_metrics.png",
                format="png",
                bbox_inches="tight",
            )
            plt.close()
