from flask import Flask, render_template, request, redirect, url_for, jsonify
import dicttoxml # Librería que parsea de JSON a XML
from xml.dom.minidom import parseString # Librería que formatea el XML para que sea legible

app = Flask(__name__)

# Simulación de Base de Datos 
tickets = [
  {'id': 1, 'titulo': 'Falla de Internet', 'estado': 'Cerrado'},
  {'id': 2, 'titulo': 'Teclado no funciona', 'estado': 'Abierto'}
]

@app.route('/')
def index():
  # 1. Cálculos para el informe
  total = len(tickets)
  resueltos = len([t for t in tickets if t['estado'] == 'Cerrado'])
  porcentaje = (resueltos / total * 100)

  # 2. Crear el diccionario de datos (JSON)
  datos_informe = {
    'resumen': {
      'total_tickets': total,
      'resueltos': resueltos,
      'porcentaje_exito': f"{porcentaje}%"
    },
    'lista_tickets': tickets
  }

  # 3. Parseo a XML
  xml_puro = dicttoxml.dicttoxml(datos_informe, custom_root='InformeSoporte', attr_type=False)

  # 4. Formatear para que se vea como árbol en el navegador
  xml_formateado = parseString(xml_puro).toprettyxml()

  return render_template('index.html', tickets=tickets, xml=xml_formateado, total=total, porcentaje=porcentaje)

# Ruta para CREAR un ticket
@app.route('/crear', methods=['POST'])
def crear():
  nuevo_id = len(tickets) + 1
  titulo = request.form.get('titulo')
  estado = request.form.get('estado')
  tickets.append({'id': nuevo_id, 'titulo': titulo, 'estado': estado})
  return redirect(url_for('index'))

@app.route('/formulario')
def mostrar_formulario():
  return render_template('formulario.html')

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=5000, debug=True)
