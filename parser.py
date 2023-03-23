file=r'dblp-2021-02-01.xml'

elements= ["knowledge representation","Representation","semantic web","semantic website","automated reasoning","declarative knowledge","procedural knowledge",
         "Meta-knowledge","Heuristic knowledge","semantic net","semantic network","semantic nets","Knowledge-based systems",
         "expert systems","Frames"]
elements= [each_string.lower() for each_string in elements]



count=0
data= dict()


with open(file, "r") as f:
    for line in f:
        line = f.readline().strip()
        if line.startswith('<title>'):
            title_start_pos = line.find("<title>")
            title_end_pos = line.find("</title>")
            title = line[title_start_pos+7: title_end_pos]
            title= title.lower()
            
            
            if any( element in title for element in elements ):
            
              
                count+=1
                line1 = next(f).strip()
                if line1.startswith('<year>'):
                    year_start_position = line1.find("<year>")
                    year_end_position=line1.find("</year>")
                    year = line1[year_start_position+6 :year_end_position]
                    
                    if year not in data:
                        data[year] = 1
                    else:
                        data[year] += 1
                    
                        
                line2 = next(f).strip()
                if line2.startswith('<year>'):
                    year_start_position = line2.find("<year>")
                    year_end_position=line2.find("</year>")
                    year = line2[year_start_position +6:year_end_position]
                    
                    if year not in data:
                        data[year] = 1
                    else:
                        data[year] += 1
                        
                        
               
                line3 = next(f).strip()
                if line3.startswith('<year>'):
                    year_start_position = line3.find("<year>")
                    year_end_position=line3.find("</year>")
                    year = line3[year_start_position +6:year_end_position]
                    
                    if year not in data:
                        data[year] = 1
                    else:
                        data[year] += 1
                        
                        
                        
                line4 = next(f).strip()
                if line4.startswith('<year>'):
                    year_start_position = line4.find("<year>")
                    year_end_position=line4.find("</year>")
                    year = line4[year_start_position +6:year_end_position]
                    
                    if year not in data:
                        data[year] = 1
                    else:
                        data[year] += 1
                        
                        
                        
                        
print(count)             
                        

f = open(r'timeseries.txt', "w")
f.write("Years" + "\t" + "publications\n")
for i in sorted(data):
    f.write((str(i) + "\t" + str(data[i]) + "\n"))
f.close()



