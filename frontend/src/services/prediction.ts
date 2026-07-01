import api from "./api";
import type { PredictionRequest, PredictionResponse } from "../types";

export async function predict(
  data: PredictionRequest
): Promise<PredictionResponse> {
  const response = await api.post<PredictionResponse>("/predict", data);
  return response.data;
}
