import axios from "axios";

const API_BASE_KEY = "rag_api_base_url";
const DEFAULT_API_BASE = "http://localhost:8000";

function getBaseUrl(): string {
  if (typeof window !== "undefined") {
    return localStorage.getItem(API_BASE_KEY) || DEFAULT_API_BASE;
  }
  return DEFAULT_API_BASE;
}

const api = axios.create({
  baseURL: getBaseUrl(),
  timeout: 60000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
  config.baseURL = getBaseUrl();
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === "ECONNABORTED") {
      return Promise.reject(new Error("Request timed out. Please try again."));
    }
    if (!error.response) {
      return Promise.reject(
        new Error("Cannot connect to server. Check your API URL.")
      );
    }
    const status = error.response.status;
    if (status === 422) {
      const detail = error.response.data?.detail;
      const message =
        typeof detail === "string"
          ? detail
          : detail?.[0]?.msg || "Validation error";
      return Promise.reject(new Error(message));
    }
    if (status >= 500) {
      return Promise.reject(
        new Error("Server error. Please try again later.")
      );
    }
    return Promise.reject(error);
  }
);

export default api;
export { API_BASE_KEY, DEFAULT_API_BASE };
