import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import Chart from 'chart.js/auto'; // Import Chart.js
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function App() {
  const [data, setData] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const chartRef = useRef(null);
  const chartRef2 = useRef(null);
  const chartRef3 = useRef(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get(`http://localhost:8080/data`, {
          params: {
            model: selectedModel,
            date: selectedDate
          }
        });
        setData(response.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    if (selectedModel && selectedDate) {
      fetchData();
    }
  }, [selectedModel, selectedDate]);

  useEffect(() => {
    if (data) {
      // Destroy previous chart instance
      if (chartRef.current !== null) {
        chartRef.current.destroy();
      }
      // Destroy previous chart instance for the second chart
      if (chartRef2.current !== null) {
        chartRef2.current.destroy();
      }
      // Destroy previous chart instance for the second chart
      if (chartRef3.current !== null) {
        chartRef3.current.destroy();
      }
      // Render new charts
      renderChart();
      renderChart_();
      renderChart__();
    }
  }, [data]);

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleDateChange = (event) => {
    setSelectedDate(event.target.value);
  };

  const renderChart = () => {
    const ctx = document.getElementById('myChart');
    chartRef.current = new Chart(ctx, {
      type: selectedModel === 'SVR' ? 'scatter' : 'line',
      data: {
        labels: data.formatted_actual_dates,
        datasets: selectedModel === 'SVR' ? [
          {
            label: 'Predicted Values',
            data: data.prediction_values.map((value, index) => ({ x: index, y: value })),
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
            fill: false,
            pointRadius: 5 // Increase point size for better visibility
          },
          {
            label: 'Test Values',
            data: data.test_values.map((value, index) => ({ x: index, y: value })),
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1,
            fill: false,
            pointRadius: 5 // Increase point size for better visibility
          }
        ] : [
          {
            label: 'Predicted Values',
            data: [...Array(data.formatted_actual_dates.length - data.prediction_values.length).fill(null), ...data.prediction_values],
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1,
            fill: false,
          },
          {
            label: 'Test Values',
            data: [...Array(data.formatted_actual_dates.length - data.test_values.length).fill(null), ...data.test_values],
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1,
            fill: false,
          },
          {
            label: 'Training Values',
            data: [...Array(data.formatted_actual_dates.length - data.actual_values.length).fill(null), ...data.actual_values.map(item => item.smoothed)],
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false,
          }
        ]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Predicted VS Test Values',
            font: {
              size: 20
            }
          },
        }
      }
    });
  };

  const renderChart_ = () => {
    const ctx = document.getElementById('myChart1');
    chartRef2.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.formatted_actual_dates,
        datasets: [
          {
            label: 'Training Values',
            data: data.actual_values.map(item => item.smoothed),
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false,
          }
        ]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Original Data',
            font: {
              size: 20
            }
          }
        }
      }
    });
  };

  const renderChart__ = () => {
    const ctx = document.getElementById('myChart2');
    chartRef3.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.future_dates,
        datasets: [
          {
            label: 'Forecasted',
            data: data.forecast,
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false,
          }
        ]
      },
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Forecast',
            font: {
              size: 20
            }
          }
        }
      }
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
      <h1 style={{
        fontSize: '2rem', // Slightly larger font size
        fontFamily: 'Roboto, sans-serif', // Modern font
        color: '#ffffff', // Darker text color for contrast
        backgroundColor: '#0070f3', // Light background color
        padding: '12px', // Padding around the text
        borderRadius: '10px', // Rounded corners
        display: 'inline-block', 
        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
        marginBottom: '0.3rem',
        marginTop: '0.18rem'
      }}>Time Series Data Models</h1>
      <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center', width: '100%', flexWrap: 'wrap'}}>
        <div style={{ width: '100%', maxWidth: '45%', padding: '0.75rem'}}>
          <div style={{ border: '1px solid #D1D5DB', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', borderRadius: '0.375rem', padding: '1rem'}}>
            <div style={{ marginBottom: '0.5rem' }}>
              <label style={{ marginBottom: '0.5rem', display: 'block' }}>Select Model:</label>
              <select style={{ width: '100%', border: '1px solid #D1D5DB', borderRadius: '0.375rem', padding: '0.5rem' }} value={selectedModel} onChange={handleModelChange}>
                <option value="">Choose from Below</option>
                <option value="ARIMA">ARIMA</option>
                <option value="SARIMA">SARIMA</option>
                <option value="SES">SES</option>
                <option value="Prophet">Prophet</option>
                <option value="SVR">SVR</option>
                <option value="LSTM">LSTM</option>
                <option value="ANN">ANN</option>
                <option value="Hybrid">Hybrid</option>
              </select>
              <label style={{ marginTop: '1rem', marginBottom: '0.5rem', display: 'block' }}>Select Date:</label>
              <select style={{ width: '100%', border: '1px solid #D1D5DB', borderRadius: '0.375rem', padding: '0.5rem' }} value={selectedDate} onChange={handleDateChange}>
                <option value="">Choose from Below</option>
                <option value="20">20 Days</option>
                <option value="30">30 Days</option>
                <option value="40">40 Days</option>
                <option value="50">50 Days</option>
              </select>
            </div>
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'flex-start', width: '100%', flexWrap: 'wrap' }}>
        <div style={{ width: '100%', maxWidth: '45%', padding: '0.80rem', height: '400px' }}>
          {data && (
            <div style={{ border: '1px solid #D1D5DB', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', borderRadius: '0.375rem', padding: '1rem', marginBottom: '1rem', height: '100%' }}>
                <canvas id="myChart" />
            </div>
          )}
        </div>
        <div style={{ width: '100%', maxWidth: '45%', padding: '0.75rem', marginBottom: '1.5rem', height: '400px', marginLeft: '6rem' }}>
          {data?.future_dates? (
            <div style={{ border: '1px solid #D1D5DB', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', borderRadius: '0.375rem', padding: '1rem', marginBottom: '1rem', height: '100%' }}>
                <canvas id="myChart2" />
            </div>
          ): (
            <div style={{ textAlign: 'center', padding: '1rem' }}>
                <p>No future dates data available.</p>
            </div>
          )}
        </div>
        <div style={{ width: '100%', maxWidth: '45%', padding: '0.75rem', marginBottom: '1.5rem', height: '400px'}}>
          {data && (
            <div style={{ border: '1px solid #D1D5DB', boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)', borderRadius: '0.375rem', padding: '1rem', marginBottom: '1rem', height: '100%' }}>
                <canvas id="myChart1" />
            </div>
          )}
        </div>
      </div>
      {data && (
        <div style={{
          position: 'absolute', 
          top: '0', 
          right: '0', 
          padding: '1rem', 
          backgroundColor: 'linear-gradient(to right, #0070f3, #00d2ff)', // Gradient background
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)', 
          borderRadius: '15px', // Rounded corners
          fontFamily: 'Roboto, sans-serif', // Modern font
          fontSize: '1rem', 
          color: '#ffffff', 
          border: '2px solid #00d2ff' // Border color matching the gradient
        }}>
          <h3 style={{ fontSize: '1.125rem', marginBottom: '0.5rem' }}>Additional Metrics:</h3>
          <p>MAE: {data.mae_values}</p>
          <p>MSE: {data.mse_values}</p>
          <p>RMSE: {data.rmse_values}</p>
          <p>R^2: {data.r2_values}</p>
        </div>
      )}
    </div>
  );       
}

export default App;
