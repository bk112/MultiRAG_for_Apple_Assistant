document.addEventListener('DOMContentLoaded', () => {
    const queryForm = document.getElementById('query-form');
    const queryText = document.getElementById('query-text');
    const queryImage = document.getElementById('query-image');
    const imagePreview = document.getElementById('image-preview');
    const submitButton = document.getElementById('submit-button');

    const resultsArea = document.getElementById('results-area');
    const statusMessage = document.getElementById('status-message');
    const generationSection = document.getElementById('generation-section');
    const generatedResponseDiv = document.getElementById('generated-response');
    const retrievalSection = document.getElementById('retrieval-section');
    const retrievedDocsDiv = document.getElementById('retrieved-docs');
    const systemStatusDiv = document.getElementById('system-status');

    // Function to update status message display
    function updateStatus(message, type = 'processing') {
        statusMessage.textContent = message;
        statusMessage.className = `status-box status-${type}`; // Add type class
        statusMessage.classList.remove('hidden');
        console.log(`Status Update [${type}]: ${message}`);
    }

    // Function to update system status display
    function updateSystemStatus(message, type = 'pending') {
         systemStatusDiv.textContent = `System Status: ${message}`;
         systemStatusDiv.className = `status-box status-${type}`;
         console.log(`System Status Update [${type}]: ${message}`);
    }


    // Function to clear results
    function clearResults() {
        generatedResponseDiv.innerHTML = '';
        retrievedDocsDiv.innerHTML = '';
        generationSection.classList.add('hidden');
        retrievalSection.classList.add('hidden');
        statusMessage.classList.add('hidden');
        statusMessage.textContent = '';
        statusMessage.className = 'status-box'; // Reset class
    }

    // Image preview handler
    queryImage.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '#';
            imagePreview.classList.add('hidden');
        }
    });

    // Check system status on load
    async function checkSystemStatus() {
        updateSystemStatus("Checking...", "pending");
        try {
            const response = await fetch('/status');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log("System Status Response:", data);

            if (data.status === "success") {
                updateSystemStatus("Ready", "success");
                submitButton.disabled = false;
            } else if (data.status === "partial") {
                 const details = [];
                 if(data.indexer !== 'success') details.push('Indexer issue');
                 if(data.retriever !== 'success') details.push('Retriever issue');
                 if(data.generator !== 'success') details.push('Generator issue');
                 updateSystemStatus(`Partial (${details.join(', ')})`, "warning");
                 // Decide if partial is usable - maybe disable submit if retriever is down?
                 submitButton.disabled = (data.retriever === 'failed' || data.retriever === 'skipped (indexer failed)');
                 if(submitButton.disabled) {
                     updateStatus("System partially initialized. Querying disabled due to critical component failure (Retriever).", "error");
                 } else {
                      updateStatus("System partially initialized. Some features (like generation) might be unavailable.", "warning");
                 }
            }
             else { // failed or other status
                updateSystemStatus(`Failed (${data.error_message || 'Unknown reason'})`, "error");
                updateStatus(`System initialization failed: ${data.error_message || 'Unknown reason'}. Querying disabled.`, "error");
                submitButton.disabled = true;
            }
        } catch (error) {
            console.error("Error checking system status:", error);
            updateSystemStatus(`Error checking status (${error.message})`, "error");
            updateStatus(`Could not verify system status: ${error.message}. Querying disabled.`, "error");
            submitButton.disabled = true;
        }
    }

    // Form submission handler
    queryForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission
        clearResults();

        const text = queryText.value.trim();
        const imageFile = queryImage.files[0];

        if (!text && !imageFile) {
            updateStatus("Please enter a text query or upload an image.", 'error');
            return;
        }

        updateStatus("Processing your query...", 'processing');
        submitButton.disabled = true;
        resultsArea.classList.remove('hidden'); // Show results area

        const formData = new FormData();
        if (text) {
            formData.append('query_text', text);
        }
        if (imageFile) {
            formData.append('query_image', imageFile);
        }

        try {
            const response = await fetch('/query', {
                method: 'POST',
                body: formData,
                // 'Content-Type' header is automatically set by browser for FormData
            });

            if (!response.ok) {
                // Try to get error detail from response body
                let errorDetail = `HTTP error! status: ${response.status}`;
                try {
                     const errorData = await response.json();
                     errorDetail = errorData.detail || JSON.stringify(errorData);
                } catch (jsonError) {
                    // If response is not JSON, use status text
                    errorDetail = response.statusText || errorDetail;
                }
                throw new Error(errorDetail);
            }

            const data = await response.json();
            console.log("Received data:", data);

            // Display results
            displayResults(data);
            updateStatus("Query processed successfully!", 'success');

        } catch (error) {
            console.error("Error submitting query:", error);
            updateStatus(`Error: ${error.message}`, 'error');
            resultsArea.classList.remove('hidden'); // Ensure results area is visible to show error
            generationSection.classList.add('hidden');
            retrievalSection.classList.add('hidden');
        } finally {
            submitButton.disabled = false; // Re-enable button
        }
    });

    // Function to display results in the HTML
    function displayResults(data) {
        // Display Generated Response
        if (data.generated_response) {
            generatedResponseDiv.textContent = data.generated_response;
            generationSection.classList.remove('hidden');
        } else {
            generatedResponseDiv.textContent = 'No response generated.';
            generationSection.classList.remove('hidden'); // Show section even if empty? Or hide? Let's show.
        }

        // Display Retrieved Documents
        if (data.retrieved_docs && data.retrieved_docs.length > 0) {
            retrievedDocsDiv.innerHTML = ''; // Clear previous docs
            data.retrieved_docs.forEach(doc => {
                const docCard = document.createElement('div');
                docCard.classList.add('doc-card');

                const score = typeof doc.score === 'number' ? doc.score.toFixed(4) : doc.score || 'N/A';
                const textPreview = doc.text ? (doc.text.substring(0, 150) + (doc.text.length > 150 ? '...' : '')) : 'No text content available.';
                const imagePath = doc.image_path ? `Associated Image: ${doc.image_path.split(/[\\/]/).pop()}` : ''; // Show only filename

                docCard.innerHTML = `
                    <p><strong class="doc-id">ID:</strong> ${doc.id || 'N/A'}</p>
                    <p><strong class="doc-score">Score:</strong> ${score}</p>
                    <p class="doc-text"><strong>Text:</strong> ${textPreview}</p>
                    ${imagePath ? `<p class="doc-image-path">${imagePath}</p>` : ''}
                `;
                // Potential TODO: Add clickable link or small thumbnail for image_path if exists
                retrievedDocsDiv.appendChild(docCard);
            });
            retrievalSection.classList.remove('hidden');
        } else {
            retrievedDocsDiv.innerHTML = '<p>No relevant documents were retrieved.</p>';
            retrievalSection.classList.remove('hidden'); // Show the section with the message
        }

        // Display top-level error if present
         if(data.error) {
             updateStatus(`Processing finished with errors: ${data.error}`, 'error');
         }
    }

    // Initial check
    checkSystemStatus();
});