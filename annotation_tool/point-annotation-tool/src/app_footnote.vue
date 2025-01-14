<template>
    <div class="container">
      <h1>Point Annotation Tool</h1>
      <div class="controls">
        <div class="page-info">
          <span>Current Page: {{ currentPage }}</span>
          <span class="label-info">Current Label: {{ displayCurrentLabel }}</span>
          <button @click="saveLabelsSwitchPage">Save & Next Page</button>
          <button @click="toggleFootnoteMode" :class="{ active: isFootnoteMode }">
            Footnote Mode (f)
          </button>
        </div>
        <div class="instructions">
          <p>Hold Ctrl and hover over points to annotate them. Release Ctrl to move to next label.</p>
          <p>Click 'Footnote Mode' or press 'F' key to toggle footnote labeling.</p>
        </div>
      </div>
      
      <div 
        class="canvas-container" 
        @mousemove="handleMouseMove"
        @keydown.d="startAnnotation"
        @keyup.d="finishAnnotation"
        @keydown.f="toggleFootnoteMode"
        tabindex="0"
      >
        <canvas ref="canvas"></canvas>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'App',
    data() {
      return {
        currentPage: 0,
        points: [],
        labels: [],
        isCtrlPressed: false,
        ctx: null,
        currentLabel: 0,
        margin: 50,
        isFootnoteMode: false
      }
    },
    computed: {
      displayCurrentLabel() {
        return this.isFootnoteMode ? 'f' : this.currentLabel.toString()
      }
    },
    mounted() {
      this.ctx = this.$refs.canvas.getContext('2d')
      this.loadPoints()
    },
    methods: {
      toggleFootnoteMode() {
        this.isFootnoteMode = !this.isFootnoteMode
      },
      
      async loadPoints() {
        async loadPoints() {
      try {
        const response = await fetch(`/test-data/pg_${this.currentPage}_points.txt`)
        const text = await response.text()
        this.points = text.trim().split('\n').map(line => {
          const [x, y] = line.split(' ').map(Number)
          return { x, y, label: null }
        })
        this.labels = new Array(this.points.length).fill(null)
        this.updateCanvasSize()
        this.drawPoints()
      } catch (error) {
        console.error('Error loading points:', error)
      }
    },
      
      updateCanvasSize() {
        const xCoords = this.points.map(p => p.x)
        const yCoords = this.points.map(p => p.y)
        const minX = Math.min(...xCoords)
        const maxX = Math.max(...xCoords)
        const minY = Math.min(...yCoords)
        const maxY = Math.max(...yCoords)
        
        let width = maxX - minX + (this.margin * 2)
        let height = maxY - minY + (this.margin * 2)
        
        const container = document.querySelector('.container')
        const maxWidth = container.clientWidth - 40
        const maxHeight = window.innerHeight - 200
        
        const scale = Math.min(
          maxWidth / width,
          maxHeight / height,
          1
        )
        
        width *= scale
        height *= scale
        
        const canvas = this.$refs.canvas
        canvas.width = width
        canvas.height = height
        
        const scaleX = (width - this.margin * 2) / (maxX - minX)
        const scaleY = (height - this.margin * 2) / (maxY - minY)
        this.points = this.points.map(point => ({
          ...point,
          x: (point.x - minX) * scaleX + this.margin,
          y: (point.y - minY) * scaleY + this.margin
        }))
      },
      
      getColorForLabel(label) {
        if (label === 'f') {
          return '#FF4444'  // Special color for footnotes
        } else {
          const hue = (Number(label) * 137.5) % 360
          return `hsl(${hue}, 70%, 50%)`
        }
      },
      
      drawPoints() {
        this.ctx.clearRect(0, 0, this.$refs.canvas.width, this.$refs.canvas.height)
        
        // Draw all points
        this.points.forEach((point, index) => {
          this.ctx.beginPath()
          this.ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
          
          if (this.labels[index] !== null) {
            this.ctx.fillStyle = this.getColorForLabel(this.labels[index])
          } else {
            this.ctx.fillStyle = 'gray'
          }
          
          this.ctx.fill()
        })
      },
      
      handleMouseMove(event) {
        if (!this.isCtrlPressed) return
        
        const rect = this.$refs.canvas.getBoundingClientRect()
        const scale = rect.width / this.$refs.canvas.width
        const x = (event.clientX - rect.left) / scale
        const y = (event.clientY - rect.top) / scale
        
        const closePoint = this.points.findIndex(point => {
          const distance = Math.sqrt(
            Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2)
          )
          return distance < 10
        })
        
        if (closePoint !== -1) {
          this.labels[closePoint] = this.isFootnoteMode ? 'f' : this.currentLabel.toString()
          this.drawPoints()
        }
      },
      
      startAnnotation() {
        this.isCtrlPressed = true
      },
      
      finishAnnotation() {
        this.isCtrlPressed = false
        if (!this.isFootnoteMode) {
          this.currentLabel += 1
        }
      },
      
      async saveLabelsSwitchPage() {
        const labelsText = this.labels.join('\n')
        console.log(`Saving to pg_${this.currentPage}_labels.txt:`)
        console.log(labelsText)
        
        this.currentPage++
        this.currentLabel = 0  // Reset label counter for new page
        await this.loadPoints()
      }
    }
  }
  </script>
  
  <style>
  .container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
  }
  
  .controls {
    margin-bottom: 20px;
  }
  
  .page-info {
    margin-bottom: 10px;
    display: flex;
    gap: 20px;
    align-items: center;
  }
  
  .label-info {
    font-weight: bold;
    color: #333;
  }
  
  .instructions {
    color: #666;
  }
  
  .canvas-container {
    border: 1px solid #ccc;
    outline: none;
  }
  
  canvas {
    display: block;
  }
  
  button {
    padding: 8px 16px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  
  button:hover {
    background-color: #45a049;
  }
  
  button.active {
    background-color: #FF4444;
  }
  </style>