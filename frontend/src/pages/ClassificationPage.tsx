import { useState } from "react";
import PredictionForm from "../components/PredictionForm";
import ResultCard from "../components/ResultCard";
import { CardSkeleton } from "../components/LoadingSkeleton";
import EmptyState from "../components/EmptyState";
import ErrorState from "../components/ErrorState";
import { predict } from "../services/prediction";
import type { PredictionResponse } from "../types";

interface ClassificationPageProps {
  onPredictionComplete?: (response: PredictionResponse, query: string) => void;
  addToast: (message: string, type: "success" | "error" | "info") => void;
}

export default function ClassificationPage({
  onPredictionComplete,
  addToast,
}: ClassificationPageProps) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (query: string) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predict({ query });
      setResult(data);
      onPredictionComplete?.(data, query);
      addToast("Classification completed successfully", "success");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred";
      setError(message);
      addToast(message, "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
          Ticket Classification
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Submit a support ticket to classify and generate a response.
        </p>
      </div>

      <PredictionForm onSubmit={handleSubmit} loading={loading} />

      <div className="mt-6">
        {loading && <CardSkeleton />}
        {error && <ErrorState message={error} onRetry={() => setError(null)} />}
        {result && !loading && <ResultCard response={result} />}
        {!result && !loading && !error && (
          <EmptyState
            title="No results yet"
            description="Fill in the form above and submit a ticket to see the classification result."
          />
        )}
      </div>
    </div>
  );
}
