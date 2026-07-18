export type Gemma4CompositeTraceModality = "image" | "video" | "audio";

export interface Gemma4CompositeTraceProfile {
  modality: Gemma4CompositeTraceModality;
  tokenTypeId: 0 | 1 | 2;
  featureOutput: "image_features" | "video_features" | "audio_features";
  featureOperationId: "composite_image_features" | "composite_video_features" | "composite_audio_features";
  scatterOperationId: "composite_image_scatter" | "composite_video_scatter" | "composite_audio_scatter";
}

/** Authoritative routing shared by capture and source-removed comparison. */
export function gemma4CompositeTraceProfile(modality: Gemma4CompositeTraceModality): Gemma4CompositeTraceProfile {
  switch (modality) {
    case "image": return { modality, tokenTypeId: 1, featureOutput: "image_features", featureOperationId: "composite_image_features", scatterOperationId: "composite_image_scatter" };
    case "video": return { modality, tokenTypeId: 2, featureOutput: "video_features", featureOperationId: "composite_video_features", scatterOperationId: "composite_video_scatter" };
    case "audio": return { modality, tokenTypeId: 0, featureOutput: "audio_features", featureOperationId: "composite_audio_features", scatterOperationId: "composite_audio_scatter" };
  }
}
