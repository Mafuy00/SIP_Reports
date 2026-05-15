const chatWindow = document.getElementById('chatWindow');
const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('sendButton');
const clearButton = document.getElementById('clearButton');
const template = document.getElementById('messageTemplate');
const promptChips = document.querySelectorAll('.prompt-chip');

function timeStamp() {
  return new Intl.DateTimeFormat('en-SG', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date());
}

function appendMessage(text, role = 'bot', isWarning = false) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  if (isWarning) {
    node.classList.add('warning');
  }
  const author = role === 'user' ? 'You' : 'Assistant';
  node.querySelector('.message-meta').textContent = `${author} • ${timeStamp()}`;
  node.querySelector('.message-text').textContent = text;
  chatWindow.appendChild(node);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendTypingIndicator() {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add('bot', 'typing');
  node.querySelector('.message-meta').textContent = 'Assistant • typing';
  node.querySelector('.message-text').innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  chatWindow.appendChild(node);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return node;
}

function appendStreamingMessage() {
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add('bot');
  node.querySelector('.message-meta').textContent = `Assistant • ${timeStamp()}`;
  node.querySelector('.message-text').textContent = '';
  chatWindow.appendChild(node);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return node;
}

function updateStreamingMessage(node, text) {
  node.querySelector('.message-text').textContent = text;
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function handleQuestion(question) {
  appendMessage(question, 'user');
  sendButton.disabled = true;
  const typingNode = appendTypingIndicator();
  const streamingNode = appendStreamingMessage();

  try {
    const result = await sendQuestionStream(question, streamingNode);
    typingNode.remove();
    updateStreamingMessage(streamingNode, result.answer);
    streamingNode.classList.toggle('warning', Boolean(result.degraded));
    if (result.warning) {
      appendMessage(result.warning, 'bot', true);
    }
  } catch (error) {
    typingNode.remove();
    streamingNode.remove();
    appendMessage(`Error: ${error.message}`, 'bot', true);
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
}

async function sendQuestionStream(question, streamingNode) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });

  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || 'Request failed');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answer = '';
  let degraded = false;
  let warning = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }

      const payload = JSON.parse(line);
      if (payload.type === 'delta') {
        answer += payload.text || '';
        updateStreamingMessage(streamingNode, answer);
      } else if (payload.type === 'done') {
        answer = payload.answer || answer;
        degraded = Boolean(payload.degraded);
        warning = payload.warning || '';
        updateStreamingMessage(streamingNode, answer);
      }
    }
  }

  if (buffer.trim()) {
    const payload = JSON.parse(buffer);
    if (payload.type === 'done') {
      answer = payload.answer || answer;
      degraded = Boolean(payload.degraded);
      warning = payload.warning || '';
      updateStreamingMessage(streamingNode, answer);
    }
  }

  return { answer, degraded, warning };
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  questionInput.value = '';
  await handleQuestion(question);
});

promptChips.forEach((chip) => {
  chip.addEventListener('click', async () => {
    const prompt = chip.dataset.prompt;
    if (!prompt || sendButton.disabled) {
      return;
    }
    questionInput.value = '';
    await handleQuestion(prompt);
  });
});

if (clearButton) {
  clearButton.addEventListener('click', () => {
    chatWindow.innerHTML = '';
    appendMessage('Chat cleared. Ask a new question when you are ready.', 'bot');
  });
}

appendMessage('Welcome! Ask a question about your SIP reports. I will answer from the retrieved context.', 'bot');
