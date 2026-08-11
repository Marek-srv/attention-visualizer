export type AppMode = "toy" | "training" | "prediction" | "trained-inspector" | "real-model";

export type AppRoute = {
  id: AppMode;
  label: string;
  shortLabel: string;
  description: string;
};

export const APP_ROUTES: readonly AppRoute[] = [
  { id: "toy", label: "Toy Math Lab", shortLabel: "Toy", description: "Transparent fixed four-dimensional calculations" },
  { id: "training", label: "Train Model", shortLabel: "Train", description: "Fit the local tiny decoder" },
  { id: "prediction", label: "Predict", shortLabel: "Predict", description: "Use locally trained weights" },
  { id: "trained-inspector", label: "Trained Model Inspector", shortLabel: "Inspector", description: "Trace one learned layer and head" },
  { id: "real-model", label: "Real Model", shortLabel: "Real", description: "Explicitly load a small pretrained model" },
] as const;

export function routeFromHash(hash: string): AppMode {
  const candidate = hash.replace(/^#\/?/, "") as AppMode;
  return APP_ROUTES.some((route) => route.id === candidate) ? candidate : "toy";
}

export function hashForMode(mode: AppMode): string {
  return `#/${mode}`;
}

