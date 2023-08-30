# Reciept-Scanner-
This is a recipet scanner based on registering of image and then doing ocr to extarct the numerical values<br>
In future I might update it to extarct all the informations like items description, date and all the information and redirect it into csv<br>
![alt text](https://github.com/vageesh1/Reciept-Scanner-/blob/main/recipt%20image.png)<br>
The result came out to be 14 as the total amount

I have also added functionality of making a PDF chatbot for invoice scanning which takes PDF as an input parse it and then I made the vector database using FAISS. <br>
The LLM that was used is LLAMAv2-7B-4bit(you can also infer it on google colab)<br>
The prompt feeding, inferencing, chaining was done by Langchain<br>
The prompt is set such that the output should be precise only

