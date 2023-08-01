package openaiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// HTTPClient is an interface that our Client and MockClient should satisfy
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// Client is the OpenAI client.
type Client struct {
	apiKey     string
	httpClient HTTPClient
}

type (

	// EmbeddingRequest is the request body for the embedding endpoint.
	EmbbedingRequest struct {
		Model string `json:"model"`
		Input string `json:"input"`
	}

	// EmbeddingResponse is the response body for the embedding endpoint.
	EmbeddingResponse struct {
		Object string      `json:"object"`
		Data   []Embedding `json:"data"`
		Model  string      `json:"model"`
		Usage  Usage       `json:"usage"`
	}

	// Embedding is the embedding data containing the embedding vector.
	Embedding struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	}

	// Usage is the token usage data.
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		TotalTokens      int `json:"total_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	}

	// CompletitionRequest is the request body for the completition endpoint.
	CompletitionRequest struct {
		Model    string    `json:"model"`
		Messages []Message `json:"messages"`
	}

	// CompletitionResponse is the response body for the completition endpoint.
	CompletitionResponse struct {
		ID      string   `json:"id"`
		Object  string   `json:"object"`
		Model   string   `json:"model"`
		Created int      `json:"created"`
		Choices []Choice `json:"choices"`
		Usage   Usage    `json:"usage"`
	}

	// CompletitionResponse is the response body for the completition endpoint.
	Choice struct {
		Index        int     `json:"index"`
		FinishReason string  `json:"finish_reason"`
		Message      Message `json:"message"`
	}

	// CompletitionResponse is the response body for the completition endpoint.
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}

	// // Client is the OpenAI client.
	// Client struct {
	// 	apiKey     string
	// 	httpClient *http.Client
	// }
)

// New creates a new OpenAI client.
func New(apiKey string, httpClient HTTPClient) *Client {
	return &Client{
		apiKey:     apiKey,
		httpClient: httpClient,
	}
}

// CreateEmbedding creates an embedding for the given text.
func (c *Client) CreateEmbedding(ctx context.Context, in EmbbedingRequest) (*EmbeddingResponse, error) {
	jsonData, err := json.Marshal(in)
	if err != nil {
		return nil, fmt.Errorf("could not marshal data: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("could not create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("could not send request: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var embResp EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("could not decode response: %w", err)
	}
	return &embResp, nil
}

// CreateChatCompletition creates a completition for the given messages.
func (c *Client) CreateChatCompletition(ctx context.Context, in CompletitionRequest) (*CompletitionResponse, error) {
	jsonData, err := json.Marshal(in)
	if err != nil {
		return nil, fmt.Errorf("could not marshal data: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("could not create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("could not send request: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var compResp CompletitionResponse
	if err := json.NewDecoder(resp.Body).Decode(&compResp); err != nil {
		return nil, fmt.Errorf("could not decode response: %w", err)
	}
	return &compResp, nil
}
