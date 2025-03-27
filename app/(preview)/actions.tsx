import { Message, TextStreamMessage } from "@/components/message";
import { openai } from "@ai-sdk/openai";
import { CoreMessage, generateId } from "ai";
import {
  createAI,
  createStreamableValue,
  getMutableAIState,
  streamUI,
} from "ai/rsc";
import { ReactNode } from "react";
import { z } from "zod";
import { CameraView } from "@/components/camera-view";
import { HubView } from "@/components/hub-view";
import { UsageView } from "@/components/usage-view";

export interface Hub {
  climate: Record<"low" | "high", number>;
  lights: Array<{ name: string; status: boolean }>;
  locks: Array<{ name: string; isLocked: boolean }>;
}

let hub: Hub = {
  climate: {
    low: 23,
    high: 25,
  },
  lights: [
    { name: "patio", status: true },
    { name: "kitchen", status: false },
    { name: "garage", status: true },
  ],
  locks: [{ name: "back door", isLocked: true }],
};

const sendMessage = async (message: string) => {
  "use server";

  // Update state with user message
  const messages = getMutableAIState<typeof AI>("messages");
  messages.update([
    ...(messages.get() as CoreMessage[]),
    { role: "user", content: message },
  ]);

  // Send the message to your Python backend
  const response = await fetch("http://127.0.0.1:8080/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: messages.get() }),
  });

  const { content } = await response.json(); // Full response

  // Create a streamable effect manually
  const contentStream = createStreamableValue("");
  const textComponent = <TextStreamMessage content={contentStream.value} />;

  // Simulate typing effect
  (async () => {
    for (let i = 0; i < content.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 20)); // Typing delay
      contentStream.update(content.slice(0, i + 1)); // Add one character at a time
    }
    contentStream.done(); // Mark stream as completed
  })();

  return textComponent;
};


export type UIState = Array<ReactNode>;

export type AIState = {
  chatId: string;
  messages: Array<CoreMessage>;
};

export const AI = createAI<AIState, UIState>({
  initialAIState: {
    chatId: generateId(),
    messages: [],
  },
  initialUIState: [],
  actions: {
    sendMessage,
  },
  onSetAIState: async ({ state, done }) => {
    "use server";

    if (done) {
      // save to database
    }
  },
});
