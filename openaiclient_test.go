package openaiclient

import (
	"bytes"
	"context"
	"encoding/json"
	"io"

	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockHTTPClient is the mock client
type mockHTTPClient struct {
	DoFunc func(req *http.Request) (*http.Response, error)
}

// Do is the mock client's `Do` function
func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	return m.DoFunc(req)
}

func TestClient_CreateEmbedding(t *testing.T) {
	t.Parallel()

	payload := EmbeddingResponse{
		Object: "test_embedding",
		Data: []Embedding{
			{
				Object:    "embedding",
				Embedding: []float32{0.1, 0.2, 0.3},
				Index:     0,
			},
		},
		Model: "test_model",
		Usage: Usage{
			PromptTokens:     1,
			TotalTokens:      2,
			CompletionTokens: 1,
		},
	}

	payloadBytes, err := json.Marshal(payload)
	require.NoError(t, err)

	tests := []struct {
		name     string
		response *http.Response
		want     *EmbeddingResponse
		wantErr  bool
	}{
		{
			name: "returns a valid embedding",
			response: &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(bytes.NewReader(payloadBytes)),
			},
			want:    &payload,
			wantErr: false,
		},
		{
			name: "returns an error if the status code is not 200",
			response: &http.Response{
				StatusCode: 500,
				Body:       io.NopCloser(bytes.NewReader(payloadBytes)),
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "returns an error if the response body is not valid json",
			response: &http.Response{
				StatusCode: 200,
				Body:       io.NopCloser(strings.NewReader("invalid json")),
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := New("test_api_key", &mockHTTPClient{
				DoFunc: func(req *http.Request) (*http.Response, error) {
					return tt.response, nil
				},
			})

			embResp, err := client.CreateEmbedding(context.Background(), EmbbedingRequest{})

			assert.Equal(t, tt.wantErr, err != nil)
			assert.Equal(t, tt.want, embResp)
		})
	}
}

func TestClient_CreateChatCompletition(t *testing.T) {
	t.Parallel()

	t.Run("client send correct headers", func(t *testing.T) {
		t.Parallel()

		client := New("test_api_key", &mockHTTPClient{
			DoFunc: func(req *http.Request) (*http.Response, error) {
				assert.Equal(t, "Bearer test_api_key", req.Header.Get("Authorization"))
				assert.Equal(t, "application/json", req.Header.Get("Content-Type"))
				return &http.Response{
					StatusCode: 200,
					Body:       io.NopCloser(strings.NewReader("{}")),
				}, nil
			},
		})

		_, err := client.CreateChatCompletition(context.Background(), CompletitionRequest{})
		require.NoError(t, err)
	})

	t.Run("client send correct parameters", func(t *testing.T) {
		t.Parallel()

		input := CompletitionRequest{
			Model: "test_model",
			Messages: []Message{
				{
					Role:    "test_role",
					Content: "test_content",
				},
			},
		}

		client := New("test_api_key", &mockHTTPClient{
			DoFunc: func(req *http.Request) (*http.Response, error) {
				assert.Equal(t, "Bearer test_api_key", req.Header.Get("Authorization"))
				assert.Equal(t, "application/json", req.Header.Get("Content-Type"))

				bodyBytes, err := io.ReadAll(req.Body)
				require.NoError(t, err)

				var body CompletitionRequest
				err = json.Unmarshal(bodyBytes, &body)
				require.NoError(t, err)

				assert.Equal(t, input, body)

				return &http.Response{
					StatusCode: 200,
					Body:       io.NopCloser(strings.NewReader("{}")),
				}, nil
			},
		})

		_, err := client.CreateChatCompletition(context.Background(), input)
		require.NoError(t, err)
	})

	t.Run("test bad status code and invalid payload", func(t *testing.T) {
		t.Parallel()

		payload := CompletitionResponse{
			ID:      "test_id",
			Object:  "test_completion",
			Model:   "test_model",
			Created: 123,
			Choices: []Choice{
				{
					Index:        0,
					FinishReason: "test_reason",
					Message: Message{
						Role:    "test_role",
						Content: "test_content",
					},
				},
			},
			Usage: Usage{
				PromptTokens:     1,
				TotalTokens:      2,
				CompletionTokens: 1,
			},
		}

		payloadBytes, err := json.Marshal(payload)
		require.NoError(t, err)

		tests := []struct {
			name     string
			response *http.Response
			want     *CompletitionResponse
			wantErr  bool
		}{
			{
				name: "returns a valid completion",
				response: &http.Response{
					StatusCode: 200,
					Body:       io.NopCloser(bytes.NewReader(payloadBytes)),
				},
				want:    &payload,
				wantErr: false,
			},
			{
				name: "returns an error if the status code is not 200",
				response: &http.Response{
					StatusCode: 500,
					Body:       io.NopCloser(bytes.NewReader(payloadBytes)),
				},
				want:    nil,
				wantErr: true,
			},
			{
				name: "returns an error if the response body is not valid json",
				response: &http.Response{
					StatusCode: 200,
					Body:       io.NopCloser(strings.NewReader("invalid json")),
				},
				want:    nil,
				wantErr: true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				client := New("test_api_key", &mockHTTPClient{
					DoFunc: func(req *http.Request) (*http.Response, error) {
						return tt.response, nil
					},
				})

				compResp, err := client.CreateChatCompletition(context.Background(), CompletitionRequest{})

				assert.Equal(t, tt.wantErr, err != nil)
				assert.Equal(t, tt.want, compResp)
			})
		}
	})
}
