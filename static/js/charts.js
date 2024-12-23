// Initialize charts
let priceChart = null;
let volumeChart = null;
let hourlyPredictionChart = null;

// Chart.js default configuration
Chart.defaults.color = '#E5E7EB';
Chart.defaults.borderColor = '#374151';

// Format number to currency
const formatCurrency = (number) => {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(number);
};

// Format large numbers
const formatNumber = (number) => {
    return new Intl.NumberFormat('en-US', {
        notation: 'compact',
        compactDisplay: 'short'
    }).format(number);
};

// Update time display
const updateTimeDisplay = () => {
    const now = new Date();
    document.getElementById('lastUpdate').textContent = now.toLocaleString();
};

// Add at the beginning of the file
let previousPrice = null;

// Function to request notification permission
const requestNotificationPermission = async () => {
    if ("Notification" in window) {
        const permission = await Notification.requestPermission();
        return permission === "granted";
    }
    return false;
};

// Function to send browser notification
const sendNotification = (price, change, type) => {
    if (Notification.permission === "granted") {
        new Notification("AAPL Stock Alert", {
            body: `${type}: $${price.toFixed(2)} (${change.toFixed(2)}%)`,
            icon: "/static/images/stock-icon.png"  // Add an icon file
        });
    }
};

// Update stats
const updateStats = async () => {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        const currentPrice = parseFloat(data.current_price);
        
        // Check for significant price movements
        if (previousPrice !== null) {
            const change = ((currentPrice - previousPrice) / previousPrice) * 100;
            
            if (change <= -1) {
                sendNotification(currentPrice, change, "Price Drop Alert");
            } else if (change >= 1) {
                sendNotification(currentPrice, change, "Price Rise Alert");
            }
        }
        
        previousPrice = currentPrice;
        
        document.getElementById('currentPrice').textContent = formatCurrency(data.current_price);
        
        const changeElement = document.getElementById('priceChange');
        changeElement.textContent = `${data.daily_change.toFixed(2)}%`;
        changeElement.className = data.daily_change >= 0 ? 'text-green-500' : 'text-red-500';
        
        document.getElementById('volume').textContent = formatNumber(data.daily_volume);
        document.getElementById('dayHigh').textContent = formatCurrency(data.daily_high);
        document.getElementById('dayLow').textContent = formatCurrency(data.daily_low);
        
        // Update prediction
        const predictionElement = document.getElementById('nextDayPrediction');
        if (data.next_day_prediction !== null) {
            predictionElement.textContent = formatCurrency(data.next_day_prediction);
            predictionElement.classList.remove('text-gray-400');
            predictionElement.classList.add('text-yellow-400');
        } else {
            predictionElement.textContent = 'Prediction not available';
            predictionElement.classList.remove('text-yellow-400');
            predictionElement.classList.add('text-gray-400');
        }
        
        updateTimeDisplay();
    } catch (error) {
        console.error('Error updating stats:', error);
    }
};

// Update charts
const updateCharts = async () => {
    try {
        const response = await fetch('/api/stock-data');
        const data = await response.json();
        
        // Debug logging
        console.log("Received data:", data);
        console.log("Predictions:", data.predictions);
        console.log("Prediction dates:", data.prediction_dates);

        // Update price chart
        if (priceChart) {
            priceChart.destroy();
        }
        
        const priceCtx = document.getElementById('priceChart').getContext('2d');
        priceChart = new Chart(priceCtx, {
            type: 'line',
            data: {
                labels: [...data.timestamps, ...data.prediction_dates],
                datasets: [
                    {
                        label: 'Historical Price',
                        data: data.prices,
                        borderColor: '#10B981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 2,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Predicted Price',
                        data: [...Array(data.prices.length).fill(null), ...data.predictions],
                        borderColor: '#FBBF24',
                        backgroundColor: 'rgba(251, 191, 36, 0.1)',
                        borderDash: [5, 5],
                        tension: 0.4,
                        fill: true,
                        pointStyle: 'circle',
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#FBBF24'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            font: {
                                size: 12
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'AAPL Stock Price with Predictions',
                        padding: {
                            top: 10,
                            bottom: 20
                        },
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    tooltip: {
                        enabled: true,
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += formatCurrency(context.parsed.y);
                                    
                                    // Add percentage change for predictions
                                    if (context.dataset.label === 'Predicted Price' && context.dataIndex > 0) {
                                        const prevValue = context.dataset.data[context.dataIndex - 1];
                                        if (prevValue) {
                                            const change = ((context.parsed.y - prevValue) / prevValue) * 100;
                                            label += ` (${change >= 0 ? '+' : ''}${change.toFixed(2)}%)`;
                                        }
                                    }
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: '#374151',
                            drawBorder: false
                        },
                        ticks: {
                            callback: function(value) {
                                return formatCurrency(value);
                            },
                            padding: 10
                        },
                        title: {
                            display: true,
                            text: 'Price (USD)',
                            padding: {
                                top: 10,
                                bottom: 10
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: '#374151',
                            drawBorder: false
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45,
                            padding: 10
                        },
                        title: {
                            display: true,
                            text: 'Time',
                            padding: {
                                top: 10,
                                bottom: 10
                            }
                        }
                    }
                }
            }
        });
        
        // Update volume chart
        if (volumeChart) {
            volumeChart.destroy();
        }
        
        const volumeCtx = document.getElementById('volumeChart').getContext('2d');
        volumeChart = new Chart(volumeCtx, {
            type: 'bar',
            data: {
                labels: data.timestamps,
                datasets: [{
                    label: 'Volume',
                    data: data.volumes,
                    backgroundColor: '#60A5FA'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    title: {
                        display: true,
                        text: 'Trading Volume'
                    }
                },
                scales: {
                    y: {
                        grid: {
                            color: '#374151'
                        }
                    },
                    x: {
                        grid: {
                            color: '#374151'
                        }
                    }
                }
            }
        });
        
    } catch (error) {
        console.error('Error updating charts:', error);
    }
};

// Update hourly predictions
const updateHourlyPredictions = async () => {
    try {
        const response = await fetch('/api/hourly-predictions');
        const data = await response.json();
        
        console.log("Hourly prediction data:", data);
        
        if (data.predictions && data.predictions.length > 0) {
            const prices = data.predictions.map(p => p.price);
            const changes = data.predictions.map(p => p.change);
            
            const trace = {
                x: data.times,
                y: prices,
                type: 'bar',
                marker: {
                    color: changes.map(change => 
                        change >= 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'
                    ),
                    line: {
                        color: changes.map(change => 
                            change >= 0 ? '#10B981' : '#EF4444'
                        ),
                        width: 1
                    }
                },
                text: changes.map(change => `${change.toFixed(2)}%`),
                textposition: 'auto',
                hovertemplate: 
                    'Time: %{x}<br>' +
                    'Price: $%{y:.2f}<br>' +
                    'Change: %{text}<br>' +
                    '<extra></extra>'
            };

            const layout = {
                title: 'Hourly Price Predictions',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(45,45,45,1)',
                height: 300,
                font: {
                    color: '#E5E7EB'
                },
                xaxis: {
                    gridcolor: '#374151',
                    title: 'Time'
                },
                yaxis: {
                    gridcolor: '#374151',
                    title: 'Price (USD)',
                    tickformat: '$.2f'
                },
                margin: {
                    l: 50,
                    r: 20,
                    t: 40,
                    b: 30
                },
                showlegend: false,
                bargap: 0.2
            };

            const config = {
                responsive: true,
                displayModeBar: false
            };

            Plotly.newPlot('hourlyPredictionChart', [trace], layout, config);
        } else {
            // Show empty state
            const layout = {
                title: 'No Predictions Available',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(45,45,45,1)',
                height: 300,
                font: {
                    color: '#E5E7EB'
                }
            };
            Plotly.newPlot('hourlyPredictionChart', [], layout);
        }
    } catch (error) {
        console.error('Error updating hourly predictions:', error);
    }
};

// Update the email handling functions
async function handleEmailSubmit(event) {
    event.preventDefault();
    const emailInput = document.getElementById('emailInput');
    const email = emailInput.value.trim();
    
    if (!email) return;
    
    try {
        console.log('Sending email:', email);
        const response = await fetch('/api/alert-receivers', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: email })
        });
        
        const data = await response.json();
        console.log('Response:', data);
        
        if (data.success) {
            emailInput.value = '';
            await loadEmailReceivers();
            showMessage('Email added successfully!', 'success');
        } else {
            showMessage(data.message, 'warning');
        }
    } catch (error) {
        console.error('Error adding email:', error);
        showMessage('Error adding email', 'error');
    }
}

// Add a function to show messages
function showMessage(message, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `alert-message ${type}`;
    messageDiv.textContent = message;
    
    // Add the message to the page
    const alertSettings = document.querySelector('.alert-settings');
    alertSettings.insertBefore(messageDiv, document.getElementById('emailList'));
    
    // Remove the message after 3 seconds
    setTimeout(() => {
        messageDiv.remove();
    }, 3000);
}

// Update the email list loading function
async function loadEmailReceivers() {
    try {
        console.log('Loading email receivers...');
        const response = await fetch('/api/alert-receivers');
        const data = await response.json();
        console.log('Loaded receivers:', data);
        
        const emailList = document.getElementById('emailList');
        emailList.innerHTML = '';
        
        if (data.receivers && data.receivers.length > 0) {
            data.receivers.forEach(email => {
                const div = document.createElement('div');
                div.className = 'email-item';
                div.innerHTML = `
                    <span>${email}</span>
                    <button onclick="removeEmailReceiver('${email}')">Remove</button>
                `;
                emailList.appendChild(div);
            });
        } else {
            emailList.innerHTML = '<div class="no-emails">No email addresses added</div>';
        }
    } catch (error) {
        console.error('Error loading email receivers:', error);
        showMessage('Error loading email list', 'error');
    }
}

// Update the remove function
async function removeEmailReceiver(email) {
    try {
        const response = await fetch(`/api/alert-receivers/${email}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        if (data.success) {
            await loadEmailReceivers();
            showMessage('Email removed successfully!', 'success');
        } else {
            showMessage('Error removing email', 'error');
        }
    } catch (error) {
        console.error('Error removing email:', error);
        showMessage('Error removing email', 'error');
    }
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', function() {
    requestNotificationPermission();
    // Initial updates
    updateStats();
    updateCharts();
    updateHourlyPredictions();
    loadEmailReceivers();  // Load initial email list
    
    // Set up periodic updates
    setInterval(updateStats, 5000);      // Update stats every 5 seconds
    setInterval(updateCharts, 30000);    // Update charts every 30 seconds
    setInterval(updateHourlyPredictions, 30000);  // Update predictions every 30 seconds
});