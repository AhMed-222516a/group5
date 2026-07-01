export interface PredictionRequest {
  query: string;
  top_k?: number;
  min_similarity?: number;
}

export interface RetrievalResult {
  rank: number;
  score: number;
  ticket_id: string;
  category: string;
  priority: string;
  subject: string;
  resolution_note: string;
  knowledge_article: string;
  metadata: Record<string, unknown>;
}

export interface PredictionResponse {
  request_id: string;
  query: string;
  answer: string;
  retrieved_context: string;
  retrieval_results: RetrievalResult[];
  retrieval_time: number;
  generation_time: number;
  total_latency: number;
  generation_model: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface HistoryItem {
  id: string;
  request: PredictionRequest;
  response: PredictionResponse;
  timestamp: number;
}
