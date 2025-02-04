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
    </div>
    
    <div 
      class="canvas-container" 
      @mousemove="handleMouseMove"
      @keydown.d="startAnnotation"
      @keyup.d="finishAnnotation"
      @keydown.f="toggleFootnoteMode"
      @keydown.ctrl.z.prevent="handleUndo"
      tabindex="0"
      ref="canvasContainer"
    >
      <canvas ref="canvas"></canvas>
    </div>
  </div>
</template>

<script>
const API_BASE_URL = 'http://localhost:5000/api';
const FOOTNOTE_LABEL = 'f';

export default {
  name: 'App',
  data() {
    return {
      currentPage: 0,
      points: [],
      labels: [],
      pointColors: [], // Track point colors
      isCtrlPressed: false,
      ctx: null,
      currentLabel: 0,
      margin: 50,
      isFootnoteMode: false,
      undoHistory: [], // Stack to store annotation actions for undo
      currentAnnotationBatch: null // Store current batch of annotations
    }
  },
  computed: {
    displayCurrentLabel() {
      return this.isFootnoteMode ? FOOTNOTE_LABEL : this.currentLabel.toString()
    }
  },
  mounted() {
    this.ctx = this.$refs.canvas.getContext('2d')
    this.loadPoints()
    this.$refs.canvasContainer.focus()

    // Add keyboard event listener for undo
    window.addEventListener('keydown', this.handleKeyDown)
  },
  beforeDestroy() {
    window.removeEventListener('keydown', this.handleKeyDown)
  },
  methods: {
    async loadPoints() {
      try {
        const response = await fetch(`${API_BASE_URL}/points/${this.currentPage}`)
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        const text = await response.text()
        this.points = text.trim().split('\n').map((line, index) => {
          const [x, y] = line.split(' ').map(Number)
          return { x, y, index, label: null } // Store original index
        })
        this.labels = new Array(this.points.length).fill(null)
        this.updateCanvasSize()
        this.drawPoints()
        this.undoHistory = [] // Clear undo history when loading new points
      } catch (error) {
        console.error('Error loading points:', error)
        alert(`Error loading points for page ${this.currentPage}. Please check if the server is running and data files exist.`)
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
    
    drawPoints() {
      this.ctx.clearRect(0, 0, this.$refs.canvas.width, this.$refs.canvas.height)
      
      this.points.forEach((point, index) => {
        this.ctx.beginPath()
        this.ctx.arc(point.x, point.y, 5, 0, Math.PI * 2)
        point.currentColor = this.ctx.fillStyle
        
        if (this.labels[point.index] !== null) {
          if (this.labels[point.index] === FOOTNOTE_LABEL) {
            this.ctx.fillStyle = 'purple' // Special color for footnotes
          } else {
            const hue = (this.labels[point.index] * 137.5) % 360
            this.ctx.fillStyle = `hsl(${hue}, 70%, 50%)`
          }
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
        const point = this.points[closePoint]
        const oldLabel = this.labels[point.index]
        const newLabel = this.isFootnoteMode ? FOOTNOTE_LABEL : this.currentLabel

        // Only add to current batch if label actually changed
        if (oldLabel !== newLabel) {
          console.log('Annotating point:', {
            pointIndex: point.index,
            oldLabel,
            newLabel,
            oldColor: this.ctx.fillStyle
          })
          this.labels[point.index] = newLabel
          
          // Add to current annotation batch
          if (this.currentAnnotationBatch === null) {
            this.currentAnnotationBatch = []
          }
          this.currentAnnotationBatch.push({
            pointIndex: point.index,
            oldLabel,
            newLabel,
            oldColor: this.pointColors[point.index] || 'gray' // Store previous color

          })
          
          this.drawPoints()
        }
      }
    },
    
    startAnnotation() {
      this.isCtrlPressed = true
      this.currentAnnotationBatch = [] // Initialize new batch
    },
    
    finishAnnotation() {
      this.isCtrlPressed = false
      if (this.currentAnnotationBatch && this.currentAnnotationBatch.length > 0) {
        this.undoHistory.push(this.currentAnnotationBatch) // Save batch to undo history
      }
      this.currentAnnotationBatch = null
      if (!this.isFootnoteMode) {
        this.currentLabel += 1 // Only increment for normal mode
      }
    },

    toggleFootnoteMode() {
      this.isFootnoteMode = !this.isFootnoteMode
    },
    
    handleUndo() {
      console.log('Undo triggered')
      console.log('Undo history length:', this.undoHistory.length)
      if (this.undoHistory.length > 0) {
        const lastBatch = this.undoHistory.pop()
        console.log('Last batch:', lastBatch)
        lastBatch.forEach(({ pointIndex, oldLabel }) => {
          console.log(`Restoring point ${pointIndex}:`, 
            `Old label: ${oldLabel}`, 
            `Old color: ${oldColor}`)
          this.labels[pointIndex] = oldLabel
          this.pointColors[pointIndex] = oldColor // Restore color
        })
        this.drawPoints()
      }
    },
    
    async saveLabelsSwitchPage() {
      try {
        // Create array of labels in original point order
        const orderedLabels = this.labels.map(label => 
          label === FOOTNOTE_LABEL ? -1 : label // Convert footnote label to -1 for backend
        )

        const response = await fetch(`${API_BASE_URL}/save-labels`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            page: this.currentPage,
            labels: orderedLabels
          })
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        await response.json()
        this.currentPage++
        this.currentLabel = 0
        this.isFootnoteMode = false
        this.undoHistory = [] // Clear undo history for new page
        await this.loadPoints()
      } catch (error) {
        console.error('Error saving labels:', error)
        alert('Error saving labels. Please check if the server is running and has write permissions.')
      }
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
</style>