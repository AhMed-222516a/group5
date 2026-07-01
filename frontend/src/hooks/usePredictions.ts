import { useCallback, useMemo } from "react";
import { useLocalStorage } from "./useLocalStorage";
import type { HistoryItem } from "../types";

const STORAGE_KEY = "rag_prediction_history";

export function usePredictions() {
  const [history, setHistory] = useLocalStorage<HistoryItem[]>(
    STORAGE_KEY,
    []
  );

  const addPrediction = useCallback(
    (item: HistoryItem) => {
      setHistory((prev) => [item, ...prev]);
    },
    [setHistory]
  );

  const deletePrediction = useCallback(
    (id: string) => {
      setHistory((prev) => prev.filter((item) => item.id !== id));
    },
    [setHistory]
  );

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, [setHistory]);

  const searchHistory = useCallback(
    (query: string) => {
      if (!query.trim()) return history;
      const q = query.toLowerCase();
      return history.filter(
        (item) =>
          item.request.query.toLowerCase().includes(q) ||
          item.response.answer.toLowerCase().includes(q)
      );
    },
    [history]
  );

  const sortedHistory = useMemo(
    () => [...history].sort((a, b) => b.timestamp - a.timestamp),
    [history]
  );

  return {
    history: sortedHistory,
    addPrediction,
    deletePrediction,
    clearHistory,
    searchHistory,
  };
}
